"""
PTB-XL Dataset  (PhysioNet v1.0.3)

Maps SCP diagnostic codes → 12 binary labels matching PTBXLModel.LABEL_NAMES.
Official fold split: folds 1–8 = train, fold 9 = val, fold 10 = test.

Label mapping is based on the PTB-XL SCP statement taxonomy.
Records with no relevant code at ≥ CONF_THRESHOLD % confidence are skipped
from supervised loss but still returned (labels all-zero).
"""

import ast
import os

import numpy as np
import pandas as pd
import scipy.signal as sps
import wfdb
from torch.utils.data import Dataset

# ── Constants ─────────────────────────────────────────────────────────────────

TARGET_FS  = 250      # Hz — matches EchoNext / model input
TARGET_LEN = 2500     # 10 s × 250 Hz
CONF_THRESHOLD = 70   # minimum SCP code confidence % to assign a label

# ── SCP code → label index ────────────────────────────────────────────────────
# Index meanings: see PTBXLModel.LABEL_NAMES

SCP_TO_LABEL: dict[str, int] = {
    # 0 — Normal
    "NORM":  0,

    # 1 — Inferior MI
    "IMI":   1, "ILMI": 1, "IPMI": 1, "IPLMI": 1, "INJMI": 1,

    # 2 — Anterior MI
    "AMI":   2, "ASMI": 2, "ALMI": 2, "AAMI": 2,

    # 3 — Lateral MI
    "LMI":   3,

    # 4 — Posterior MI
    "PMI":   4,

    # 5 — ST / T-wave changes (non-MI ischaemia & non-specific)
    "NST_":  5, "STD_": 5, "STE_": 5, "ISC_": 5,
    "ISCA":  5, "ISCI": 5, "NT_":  5, "INVT": 5,

    # 6 — LBBB
    "LBBB":  6, "CLBBB": 6,

    # 7 — RBBB
    "RBBB":  7, "CRBBB": 7, "IRBBB": 7,

    # 8 — AV block (any degree)
    "1AVB":  8, "2AVB": 8, "3AVB": 8, "IAVB": 8,

    # 9 — LV hypertrophy
    "LVH":   9, "LVOLT": 9,

    # 10 — Atrial fibrillation / flutter
    "AFIB": 10, "AFLT": 10,

    # 11 — composite MI (set programmatically from indices 1–4)
}

NUM_LABELS = 12
MI_INDICES  = {1, 2, 3, 4}   # any of these → also set label 11


class PTBXLDataset(Dataset):
    """
    Args:
        data_dir:        Root of PTB-XL download (contains ptbxl_database.csv).
        split:           'train' | 'val' | 'test'
        sampling_rate:   500 (high-res) or 100 (low-res).  500 recommended.
        augment:         Apply random time-shift augmentation (train only).
    """

    def __init__(
        self,
        data_dir: str,
        split: str = "train",
        sampling_rate: int = 500,
        augment: bool = False,
    ):
        self.data_dir  = data_dir
        self.fs_src    = sampling_rate
        self.augment   = augment

        meta = pd.read_csv(
            os.path.join(data_dir, "ptbxl_database.csv"), index_col="ecg_id"
        )
        meta["scp_codes"] = meta["scp_codes"].apply(ast.literal_eval)

        # Official split by strat_fold
        if split == "train":
            meta = meta[meta["strat_fold"] <= 8]
        elif split == "val":
            meta = meta[meta["strat_fold"] == 9]
        else:
            meta = meta[meta["strat_fold"] == 10]

        fn_col = "filename_hr" if sampling_rate == 500 else "filename_lr"
        self.filenames   = meta[fn_col].tolist()
        self.labels_list = [
            self._parse_labels(row["scp_codes"])
            for _, row in meta.iterrows()
        ]

    # ── Helpers ───────────────────────────────────────────────────────────────

    def _parse_labels(self, scp_codes: dict) -> np.ndarray:
        labels = np.zeros(NUM_LABELS, dtype=np.float32)
        any_mi = False
        for code, conf in scp_codes.items():
            if conf < CONF_THRESHOLD:
                continue
            idx = SCP_TO_LABEL.get(code)
            if idx is not None:
                labels[idx] = 1.0
                if idx in MI_INDICES:
                    any_mi = True
        if any_mi:
            labels[11] = 1.0
        return labels

    def _load_signal(self, fname: str) -> np.ndarray:
        path   = os.path.join(self.data_dir, fname)
        sig, _ = wfdb.rdsamp(path)        # (n_samples, 12) float64
        sig    = sig.astype(np.float32)

        # Resample to TARGET_FS (500 → 250 Hz or 100 → 250 Hz)
        if self.fs_src != TARGET_FS:
            n_out = int(sig.shape[0] * TARGET_FS / self.fs_src)
            sig   = sps.resample(sig, n_out, axis=0).astype(np.float32)

        # Truncate / zero-pad to exactly TARGET_LEN
        if sig.shape[0] >= TARGET_LEN:
            sig = sig[:TARGET_LEN]
        else:
            sig = np.pad(sig, ((0, TARGET_LEN - sig.shape[0]), (0, 0)))

        return sig  # (2500, 12)

    def _normalize(self, sig: np.ndarray) -> np.ndarray:
        """Per-lead z-score — identical to ECGDigitizer._normalize_echonext."""
        out = sig.copy()
        for i in range(out.shape[1]):
            std = out[:, i].std()
            if std > 1e-6:
                out[:, i] = np.clip((out[:, i] - out[:, i].mean()) / std, -5.0, 5.0)
        return out

    # ── Dataset interface ─────────────────────────────────────────────────────

    def __len__(self) -> int:
        return len(self.filenames)

    def __getitem__(self, idx: int):
        sig    = self._load_signal(self.filenames[idx])   # (2500, 12)

        # Random time-shift augmentation: roll ±10 % of signal length
        if self.augment:
            shift = np.random.randint(-250, 250)
            sig   = np.roll(sig, shift, axis=0)

        sig    = self._normalize(sig)
        wave   = sig.T.astype(np.float32)                 # (12, 2500) for model
        labels = self.labels_list[idx]
        return wave, labels

    def get_pos_weights(self) -> np.ndarray:
        """Inverse-frequency positive class weights for BCEWithLogitsLoss."""
        all_labels = np.stack(self.labels_list, axis=0)
        pos     = all_labels.sum(axis=0)
        neg     = len(all_labels) - pos
        weights = neg / np.maximum(pos, 1.0)
        return np.clip(weights, 1.0, 50.0).astype(np.float32)

    def class_counts(self) -> dict[str, int]:
        from model.ptbxl_model import PTBXLModel
        all_labels = np.stack(self.labels_list, axis=0)
        return {
            PTBXLModel.LABEL_NAMES[i]: int(all_labels[:, i].sum())
            for i in range(NUM_LABELS)
        }
