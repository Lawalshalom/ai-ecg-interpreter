"""
Train PTBXLModel on the PTB-XL v1.0.3 dataset.

Usage:
    # Step 1 — download data (one-time, ~1.8 GB):
    python training/ptbxl_download.py

    # Step 2 — train:
    python training/ptbxl_train.py

Outputs:
    checkpoints/ptbxl_best_model.pt   — best validation AUROC checkpoint
    checkpoints/ptbxl_last_model.pt   — final epoch checkpoint
    logs/ptbxl_train_log.csv          — per-epoch metrics

Expected performance (MPS, M-series Mac):
    ~15–25 min/epoch on 17k training records
    Target val mean AUROC: ≥ 0.90 after 20–30 epochs
"""

import csv
import os
import sys
import time

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import roc_auc_score
from torch.utils.data import DataLoader
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from model.ptbxl_model import PTBXLModel
from training.ptbxl_dataset import PTBXLDataset

# ── Paths ─────────────────────────────────────────────────────────────────────
PTBXL_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "data", "ptb-xl",
)
OUT_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "checkpoints"
)
LOG_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "logs"
)
os.makedirs(OUT_DIR, exist_ok=True)
os.makedirs(LOG_DIR, exist_ok=True)

# ── Hyperparameters ───────────────────────────────────────────────────────────
SAMPLING_RATE = 100       # 100 Hz records are fully downloaded (21,799 records)
                          # resampled to 250 Hz in the dataset loader
BATCH_SIZE    = 64        # safe for 8 GB shared RAM on MPS
NUM_EPOCHS    = 50
LR            = 3e-4
WEIGHT_DECAY  = 1e-4
GRAD_CLIP     = 1.0
PATIENCE      = 10        # early stopping patience
NUM_WORKERS   = 0         # MPS requires 0
SEED          = 42


def set_seed(seed: int):
    np.random.seed(seed)
    torch.manual_seed(seed)


def get_device() -> torch.device:
    if torch.backends.mps.is_available():
        print("Device: MPS (Apple GPU)")
        return torch.device("mps")
    print("Device: CPU")
    return torch.device("cpu")


def evaluate(
    model: PTBXLModel,
    loader: DataLoader,
    device: torch.device,
) -> dict:
    model.eval()
    all_probs, all_labels = [], []

    with torch.no_grad():
        for wave, labels in loader:
            wave   = wave.to(device)
            probs  = torch.sigmoid(model(wave)).cpu().numpy()
            all_probs.append(probs)
            all_labels.append(labels.numpy())

    probs_arr  = np.vstack(all_probs)
    labels_arr = np.vstack(all_labels)

    aurocs, names = [], PTBXLModel.LABEL_NAMES
    for i in range(labels_arr.shape[1]):
        n_pos = int(labels_arr[:, i].sum())
        if n_pos < 5:          # skip labels with almost no positives
            continue
        try:
            aurocs.append(roc_auc_score(labels_arr[:, i], probs_arr[:, i]))
        except Exception:
            pass

    per_class = {}
    for i, name in enumerate(names):
        n_pos = int(labels_arr[:, i].sum())
        if n_pos >= 5:
            try:
                per_class[name] = roc_auc_score(labels_arr[:, i], probs_arr[:, i])
            except Exception:
                per_class[name] = float("nan")

    return {
        "mean_auroc": float(np.mean(aurocs)) if aurocs else 0.0,
        "per_class":  per_class,
    }


def print_eval(results: dict, prefix: str = ""):
    print(f"\n{prefix}Mean AUROC: {results['mean_auroc']:.4f}")
    for name, auroc in sorted(results["per_class"].items(), key=lambda x: -x[1]):
        print(f"  {auroc:.4f}  {name}")


def train_one_epoch(
    model, loader, optimizer, scheduler, criterion, device, epoch
) -> float:
    model.train()
    total_loss = 0.0
    pbar = tqdm(loader, desc=f"Epoch {epoch:02d} [train]", leave=False)
    for wave, labels in pbar:
        wave, labels = wave.to(device), labels.to(device)
        optimizer.zero_grad()
        loss = criterion(model(wave), labels)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
        optimizer.step()
        total_loss += loss.item() * len(wave)
        pbar.set_postfix(loss=f"{loss.item():.4f}")
    scheduler.step()
    return total_loss / len(loader.dataset)


def main():
    if not os.path.exists(os.path.join(PTBXL_DIR, "ptbxl_database.csv")):
        print(
            f"PTB-XL not found at {PTBXL_DIR}\n"
            "Run first:  python training/ptbxl_download.py"
        )
        sys.exit(1)

    set_seed(SEED)
    device = get_device()

    # ── Datasets ──────────────────────────────────────────────────────────────
    print("Loading PTB-XL datasets ...")
    train_ds = PTBXLDataset(PTBXL_DIR, "train", SAMPLING_RATE, augment=True)
    val_ds   = PTBXLDataset(PTBXL_DIR, "val",   SAMPLING_RATE, augment=False)
    print(f"  Train: {len(train_ds):,}   Val: {len(val_ds):,}")

    print("\nLabel counts (train):")
    for name, cnt in train_ds.class_counts().items():
        print(f"  {cnt:5d}  {name}")

    pos_weights = torch.tensor(train_ds.get_pos_weights()).to(device)
    print(f"\nPos weights (first 3): {pos_weights[:3].tolist()}")

    train_loader = DataLoader(
        train_ds, batch_size=BATCH_SIZE, shuffle=True,
        num_workers=NUM_WORKERS, pin_memory=False,
    )
    val_loader = DataLoader(
        val_ds, batch_size=BATCH_SIZE * 2, shuffle=False,
        num_workers=NUM_WORKERS, pin_memory=False,
    )

    # ── Model ─────────────────────────────────────────────────────────────────
    model = PTBXLModel(dropout=0.2).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"\nModel: {n_params/1e6:.2f}M parameters")

    # ── Optimiser ─────────────────────────────────────────────────────────────
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=NUM_EPOCHS, eta_min=1e-6
    )
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weights)

    # ── Training loop ─────────────────────────────────────────────────────────
    best_auroc    = 0.0
    patience_ctr  = 0
    log_path      = os.path.join(LOG_DIR, "ptbxl_train_log.csv")

    with open(log_path, "w", newline="") as f:
        csv.writer(f).writerow(
            ["epoch", "train_loss", "val_mean_auroc", "lr", "elapsed_s"]
        )

    print(f"\nTraining for up to {NUM_EPOCHS} epochs (early-stop patience={PATIENCE}) ...\n")
    t_start = time.time()

    for epoch in range(1, NUM_EPOCHS + 1):
        t0         = time.time()
        train_loss = train_one_epoch(
            model, train_loader, optimizer, scheduler, criterion, device, epoch
        )
        val_res    = evaluate(model, val_loader, device)
        elapsed    = time.time() - t0

        mean_auroc = val_res["mean_auroc"]
        lr_now     = scheduler.get_last_lr()[0]

        print(
            f"Epoch {epoch:02d}/{NUM_EPOCHS}  "
            f"loss={train_loss:.4f}  "
            f"val_AUROC={mean_auroc:.4f}  "
            f"lr={lr_now:.2e}  "
            f"time={elapsed:.0f}s"
        )

        with open(log_path, "a", newline="") as f:
            csv.writer(f).writerow(
                [epoch, train_loss, mean_auroc, lr_now, elapsed]
            )

        if mean_auroc > best_auroc:
            best_auroc   = mean_auroc
            patience_ctr = 0
            torch.save(
                {
                    "epoch":            epoch,
                    "model_state_dict": model.state_dict(),
                    "val_mean_auroc":   float(mean_auroc),
                    "per_class_auroc":  {k: float(v) for k, v in val_res["per_class"].items()},
                },
                os.path.join(OUT_DIR, "ptbxl_best_model.pt"),
            )
            print(f"  --> New best! AUROC={best_auroc:.4f}")
        else:
            patience_ctr += 1
            if patience_ctr >= PATIENCE:
                print(f"\nEarly stopping at epoch {epoch}.")
                break

    torch.save(
        {"epoch": epoch, "model_state_dict": model.state_dict(),
         "val_mean_auroc": mean_auroc},
        os.path.join(OUT_DIR, "ptbxl_last_model.pt"),
    )

    total_hrs = (time.time() - t_start) / 3600
    print(f"\nTraining complete in {total_hrs:.2f} hrs.  Best val AUROC: {best_auroc:.4f}")
    print(f"Checkpoint: {OUT_DIR}/ptbxl_best_model.pt")

    # Final per-class evaluation on best model
    ckpt = torch.load(
        os.path.join(OUT_DIR, "ptbxl_best_model.pt"),
        map_location=device, weights_only=False,
    )
    model.load_state_dict(ckpt["model_state_dict"])
    final = evaluate(model, val_loader, device)
    print_eval(final, prefix="Best model — Val set: ")


if __name__ == "__main__":
    main()
