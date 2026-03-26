"""
Train the EchoNextModel on the EchoNext dataset using Apple MPS (or CPU fallback).

Usage:
    python training/train.py

Outputs:
    checkpoints/best_model.pt   — best validation AUROC checkpoint
    checkpoints/last_model.pt   — final epoch checkpoint
    checkpoints/norm_stats.npz  — waveform normalization stats for inference
    logs/train_log.csv          — per-epoch metrics
"""

import os
import sys
import csv
import time
import json

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

# Allow running from project root
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from model.architecture import EchoNextModel
from training.dataset import ECGDataset
from training.evaluate import evaluate_model, print_eval_results

# ── Paths ─────────────────────────────────────────────────────────────────────
DATA_DIR = "/Users/mac/Downloads/echonext-a-dataset-for-detecting-echocardiogram-confirmed-structural-heart-disease-from-ecgs-1.1.0"
OUT_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "checkpoints")
LOG_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "logs")
os.makedirs(OUT_DIR, exist_ok=True)
os.makedirs(LOG_DIR, exist_ok=True)

WAVEFORM_TRAIN = os.path.join(DATA_DIR, "EchoNext_train_waveforms.npy")
TABULAR_TRAIN  = os.path.join(DATA_DIR, "EchoNext_train_tabular_features.npy")
WAVEFORM_VAL   = os.path.join(DATA_DIR, "EchoNext_val_waveforms.npy")
TABULAR_VAL    = os.path.join(DATA_DIR, "EchoNext_val_tabular_features.npy")
METADATA       = os.path.join(DATA_DIR, "echonext_metadata_100k.csv")

# ── Hyperparameters ───────────────────────────────────────────────────────────
BATCH_SIZE     = 32
NUM_EPOCHS     = 30
LR             = 3e-4
WEIGHT_DECAY   = 1e-4
GRAD_CLIP      = 1.0
PATIENCE       = 7          # early stopping patience (epochs)
NUM_WORKERS    = 0          # macOS MPS requires 0 (spawn workers hang with MPS)
SEED           = 42


def set_seed(seed: int):
    np.random.seed(seed)
    torch.manual_seed(seed)


def get_device() -> torch.device:
    if torch.backends.mps.is_available():
        print("Using device: MPS (Apple GPU)")
        return torch.device("mps")
    print("Using device: CPU")
    return torch.device("cpu")


def save_norm_stats(train_dataset: ECGDataset):
    """
    Compute and save waveform mean/std for inference-time normalization.
    EchoNext waveforms are already normalized, but we save tabular stats too.
    """
    tab = train_dataset.tabular
    stats = {
        "tab_mean": tab.mean(axis=0).tolist(),
        "tab_std":  tab.std(axis=0).tolist(),
    }
    np.savez(os.path.join(OUT_DIR, "norm_stats.npz"), **{k: np.array(v) for k, v in stats.items()})
    with open(os.path.join(OUT_DIR, "norm_stats.json"), "w") as f:
        json.dump(stats, f, indent=2)
    print(f"Saved normalization stats to {OUT_DIR}/norm_stats.json")


def train_one_epoch(
    model: EchoNextModel,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    scheduler,
    criterion: nn.BCEWithLogitsLoss,
    device: torch.device,
    epoch: int,
) -> float:
    model.train()
    total_loss = 0.0
    pbar = tqdm(loader, desc=f"Epoch {epoch:02d} [train]", leave=False)

    for wave, tab, labels in pbar:
        wave   = wave.to(device)
        tab    = tab.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        logits = model(wave, tab)
        loss   = criterion(logits, labels)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
        optimizer.step()

        total_loss += loss.item() * len(wave)
        pbar.set_postfix(loss=f"{loss.item():.4f}")

    scheduler.step()
    return total_loss / len(loader.dataset)


def main():
    set_seed(SEED)
    device = get_device()

    # ── Datasets ──────────────────────────────────────────────────────────────
    print("Loading datasets...")
    train_ds = ECGDataset(WAVEFORM_TRAIN, TABULAR_TRAIN, METADATA, split="train", augment=True)
    val_ds   = ECGDataset(WAVEFORM_VAL,   TABULAR_VAL,   METADATA, split="val",   augment=False)

    print(f"  Train: {len(train_ds):,} samples")
    print(f"  Val:   {len(val_ds):,} samples")

    # Save normalization stats for inference
    save_norm_stats(train_ds)

    # ── Class weights ──────────────────────────────────────────────────────────
    pos_weights = train_ds.get_pos_weights().to(device)
    print(f"  Pos weights (first 3): {pos_weights[:3].tolist()}")

    # ── Loaders ───────────────────────────────────────────────────────────────
    train_loader = DataLoader(
        train_ds, batch_size=BATCH_SIZE, shuffle=True,
        num_workers=NUM_WORKERS, pin_memory=False,
    )
    val_loader = DataLoader(
        val_ds, batch_size=BATCH_SIZE * 2, shuffle=False,
        num_workers=NUM_WORKERS, pin_memory=False,
    )

    # ── Model ─────────────────────────────────────────────────────────────────
    model = EchoNextModel(dropout=0.2).to(device)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"\nModel: {total_params/1e6:.2f}M parameters")

    # ── Optimizer & scheduler ─────────────────────────────────────────────────
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=NUM_EPOCHS, eta_min=1e-6)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weights)

    # ── Training loop ─────────────────────────────────────────────────────────
    best_auroc = 0.0
    patience_counter = 0
    log_path = os.path.join(LOG_DIR, "train_log.csv")

    with open(log_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["epoch", "train_loss", "val_mean_auroc", "val_shd_auroc", "lr", "elapsed_s"])

    print(f"\nStarting training for up to {NUM_EPOCHS} epochs (early stop patience={PATIENCE})...\n")
    t_start = time.time()

    for epoch in range(1, NUM_EPOCHS + 1):
        t0 = time.time()
        train_loss = train_one_epoch(model, train_loader, optimizer, scheduler, criterion, device, epoch)
        val_results = evaluate_model(model, val_loader, device)
        elapsed = time.time() - t0

        mean_auroc = val_results["mean_auroc"]
        shd_auroc  = val_results["shd_auroc"]
        current_lr = scheduler.get_last_lr()[0]

        print(
            f"Epoch {epoch:02d}/{NUM_EPOCHS}  "
            f"loss={train_loss:.4f}  "
            f"val_mean_AUROC={mean_auroc:.4f}  "
            f"val_SHD_AUROC={shd_auroc:.4f}  "
            f"lr={current_lr:.2e}  "
            f"time={elapsed:.0f}s"
        )

        # Log
        with open(log_path, "a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([epoch, train_loss, mean_auroc, shd_auroc, current_lr, elapsed])

        # Save best
        if mean_auroc > best_auroc:
            best_auroc = mean_auroc
            patience_counter = 0
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "val_mean_auroc": mean_auroc,
                    "val_shd_auroc": shd_auroc,
                },
                os.path.join(OUT_DIR, "best_model.pt"),
            )
            print(f"  --> New best! Saved checkpoint (AUROC={best_auroc:.4f})")
        else:
            patience_counter += 1
            if patience_counter >= PATIENCE:
                print(f"\nEarly stopping at epoch {epoch} (no improvement for {PATIENCE} epochs).")
                break

    # Save final
    torch.save(
        {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "val_mean_auroc": mean_auroc,
        },
        os.path.join(OUT_DIR, "last_model.pt"),
    )

    total_time = (time.time() - t_start) / 3600
    print(f"\nTraining complete in {total_time:.2f} hrs. Best val mean AUROC: {best_auroc:.4f}")
    print(f"Best model saved to: {OUT_DIR}/best_model.pt")

    # Print final detailed eval on best model
    print("\nLoading best model for final evaluation...")
    ckpt = torch.load(os.path.join(OUT_DIR, "best_model.pt"), map_location=device)
    model.load_state_dict(ckpt["model_state_dict"])
    final_results = evaluate_model(model, val_loader, device)
    print_eval_results(final_results, prefix="Best model — Val set: ")


if __name__ == "__main__":
    main()
