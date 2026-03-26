import numpy as np
import torch
from sklearn.metrics import roc_auc_score, roc_curve

from model.architecture import EchoNextModel

LABEL_DISPLAY = EchoNextModel.LABEL_DISPLAY


def compute_auroc(y_true: np.ndarray, y_score: np.ndarray) -> np.ndarray:
    """
    Compute per-label AUROC. Returns NaN for labels with only one class present.
    y_true, y_score: (N, 12)
    """
    n_labels = y_true.shape[1]
    aurocs = np.full(n_labels, np.nan)
    for i in range(n_labels):
        if len(np.unique(y_true[:, i])) > 1:
            aurocs[i] = roc_auc_score(y_true[:, i], y_score[:, i])
    return aurocs


def compute_sensitivity_at_specificity(
    y_true: np.ndarray, y_score: np.ndarray, target_specificity: float = 0.90
) -> np.ndarray:
    """
    Compute sensitivity at target specificity for each label.
    Returns NaN for labels with only one class.
    """
    n_labels = y_true.shape[1]
    sensitivities = np.full(n_labels, np.nan)
    for i in range(n_labels):
        if len(np.unique(y_true[:, i])) < 2:
            continue
        fpr, tpr, _ = roc_curve(y_true[:, i], y_score[:, i])
        specificity = 1.0 - fpr
        # Find the point where specificity >= target and take the max sensitivity there
        mask = specificity >= target_specificity
        if mask.any():
            sensitivities[i] = tpr[mask].max()
    return sensitivities


def evaluate_model(
    model: EchoNextModel,
    loader: torch.utils.data.DataLoader,
    device: torch.device,
) -> dict:
    """
    Run full evaluation on a data loader.
    Returns dict with per-label and mean AUROC, sensitivity @ 90% specificity.
    """
    model.eval()
    all_labels, all_probs = [], []

    with torch.no_grad():
        for wave, tab, labels in loader:
            wave = wave.to(device)
            tab = tab.to(device)
            probs = torch.sigmoid(model(wave, tab)).cpu().numpy()
            all_probs.append(probs)
            all_labels.append(labels.numpy())

    all_labels = np.vstack(all_labels)   # (N, 12)
    all_probs = np.vstack(all_probs)     # (N, 12)

    aurocs = compute_auroc(all_labels, all_probs)
    sens90 = compute_sensitivity_at_specificity(all_labels, all_probs, 0.90)

    results = {
        "mean_auroc": float(np.nanmean(aurocs)),
        "shd_auroc": float(aurocs[11]),   # composite SHD label
        "per_label": {},
    }

    for i, name in enumerate(LABEL_DISPLAY):
        results["per_label"][name] = {
            "auroc": float(aurocs[i]) if not np.isnan(aurocs[i]) else None,
            "sensitivity_at_90spec": float(sens90[i]) if not np.isnan(sens90[i]) else None,
        }

    return results


def print_eval_results(results: dict, prefix: str = ""):
    print(f"\n{prefix}Mean AUROC: {results['mean_auroc']:.4f}  |  Composite SHD AUROC: {results['shd_auroc']:.4f}")
    print(f"{'Condition':<45} {'AUROC':>7}  {'Sens@90Spec':>11}")
    print("-" * 67)
    for name, metrics in results["per_label"].items():
        auroc = f"{metrics['auroc']:.4f}" if metrics["auroc"] is not None else "  N/A "
        sens = f"{metrics['sensitivity_at_90spec']:.4f}" if metrics["sensitivity_at_90spec"] is not None else "  N/A "
        print(f"{name:<45} {auroc:>7}  {sens:>11}")
