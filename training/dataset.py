import os
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

LABEL_COLS = [
    "lvef_lte_45_flag",
    "lvwt_gte_13_flag",
    "aortic_stenosis_moderate_or_greater_flag",
    "aortic_regurgitation_moderate_or_greater_flag",
    "mitral_regurgitation_moderate_or_greater_flag",
    "tricuspid_regurgitation_moderate_or_greater_flag",
    "pulmonary_regurgitation_moderate_or_greater_flag",
    "rv_systolic_dysfunction_moderate_or_greater_flag",
    "pericardial_effusion_moderate_large_flag",
    "pasp_gte_45_flag",
    "tr_max_gte_32_flag",
    "shd_moderate_or_greater_flag",
]

TABULAR_COLS = [
    "sex",
    "ventricular_rate",
    "atrial_rate",
    "pr_interval",
    "qrs_duration",
    "qt_corrected",
    "age_at_ecg",
]


class ECGDataset(Dataset):
    """
    PyTorch Dataset for EchoNext ECG data.

    Loads waveforms and tabular features from pre-split .npy files.
    Waveform shape: (N, 1, 2500, 12) -> reshaped to (12, 2500) per sample.
    Tabular shape:  (N, 7)
    Labels:         12 binary flags from metadata CSV.
    """

    def __init__(
        self,
        waveform_path: str,
        tabular_path: str,
        metadata_path: str,
        split: str,
        augment: bool = False,
    ):
        self.augment = augment

        # Check for pre-converted float32 version (half the I/O size — much faster)
        f32_path = waveform_path.replace(".npy", "_f32.npy")
        # Also check in the project data/ folder
        f32_data_path = os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            "data",
            os.path.basename(f32_path).replace(
                "EchoNext_train_waveforms", "train_waveforms"
            ).replace(
                "EchoNext_val_waveforms", "val_waveforms"
            ).replace(
                "EchoNext_test_waveforms", "test_waveforms"
            ),
        )
        if os.path.exists(f32_data_path):
            self._waveform_mmap = np.load(f32_data_path, mmap_mode="r")
            self._is_f32 = True
        elif os.path.exists(f32_path):
            self._waveform_mmap = np.load(f32_path, mmap_mode="r")
            self._is_f32 = True
        else:
            # Fallback: use float64 mmap (slow but correct)
            self._waveform_mmap = np.load(waveform_path, mmap_mode="r")
            self._is_f32 = False

        # Load tabular features: (N, 7)
        self.tabular = np.load(tabular_path).astype(np.float32)  # already standardized

        # Load labels from metadata
        meta = pd.read_csv(metadata_path, index_col=0)
        split_meta = meta[meta["split"] == split].reset_index(drop=True)

        # Labels — fill NaN as 0 (no confirmed SHD)
        labels = split_meta[LABEL_COLS].fillna(0).values.astype(np.float32)
        self.labels = labels

        assert len(self._waveform_mmap) == len(self.tabular) == len(self.labels), (
            f"Size mismatch: waveforms={len(self._waveform_mmap)}, "
            f"tabular={len(self.tabular)}, labels={len(self.labels)}"
        )

    def __len__(self):
        return len(self._waveform_mmap)

    def __getitem__(self, idx):
        # Load one sample from mmap: (1, 2500, 12) -> (12, 2500) float32
        wave = self._waveform_mmap[idx, 0, :, :].T.astype(np.float32)  # (12, 2500)
        tab = self.tabular[idx].copy()       # (7,)
        label = self.labels[idx]             # (12,)

        if self.augment:
            wave = self._augment(wave)

        return (
            torch.from_numpy(wave),
            torch.from_numpy(tab),
            torch.from_numpy(label),
        )

    def _augment(self, wave: np.ndarray) -> np.ndarray:
        """Lightweight augmentations that preserve clinical features."""
        # Gaussian noise (very small — std ~1% of signal range)
        if np.random.rand() < 0.5:
            noise = np.random.randn(*wave.shape).astype(np.float32) * 0.02
            wave = wave + noise

        # Amplitude scaling ±10%
        if np.random.rand() < 0.5:
            scale = np.random.uniform(0.9, 1.1)
            wave = wave * scale

        # Baseline wander (low-frequency drift)
        if np.random.rand() < 0.3:
            t = np.linspace(0, 2 * np.pi, wave.shape[1], dtype=np.float32)
            freq = np.random.uniform(0.1, 0.5)
            drift = np.sin(freq * t) * np.random.uniform(0.01, 0.05)
            wave = wave + drift[np.newaxis, :]

        return wave

    def get_pos_weights(self) -> torch.Tensor:
        """
        Compute per-label positive class weights for weighted BCE loss.
        pos_weight = (N_neg / N_pos) per label.
        """
        n = len(self.labels)
        pos = np.nansum(self.labels, axis=0)
        neg = n - pos
        # Clip to avoid division by zero; cap weight at 20 to prevent instability
        weights = np.clip(neg / np.maximum(pos, 1), 1.0, 20.0)
        return torch.tensor(weights, dtype=torch.float32)
