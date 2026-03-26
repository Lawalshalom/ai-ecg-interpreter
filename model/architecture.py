import torch
import torch.nn as nn
import torch.nn.functional as F


class ResBlock1D(nn.Module):
    """Residual block for 1D ECG signals."""

    def __init__(self, channels: int, kernel_size: int = 7, dropout: float = 0.1):
        super().__init__()
        padding = kernel_size // 2
        self.net = nn.Sequential(
            nn.Conv1d(channels, channels, kernel_size, padding=padding, bias=False),
            nn.BatchNorm1d(channels),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Conv1d(channels, channels, kernel_size, padding=padding, bias=False),
            nn.BatchNorm1d(channels),
        )
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.relu(self.net(x) + x)


class DownsampleBlock1D(nn.Module):
    """Strided convolution block for downsampling with residual connection."""

    def __init__(self, in_ch: int, out_ch: int, kernel_size: int = 7, stride: int = 2):
        super().__init__()
        padding = kernel_size // 2
        self.net = nn.Sequential(
            nn.Conv1d(in_ch, out_ch, kernel_size, stride=stride, padding=padding, bias=False),
            nn.BatchNorm1d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv1d(out_ch, out_ch, kernel_size, padding=padding, bias=False),
            nn.BatchNorm1d(out_ch),
        )
        self.skip = nn.Sequential(
            nn.Conv1d(in_ch, out_ch, 1, stride=stride, bias=False),
            nn.BatchNorm1d(out_ch),
        )
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.relu(self.net(x) + self.skip(x))


class ECGEncoder1D(nn.Module):
    """
    1D ResNet encoder for 12-lead ECG waveforms.

    Input:  (batch, 12, 2500)  — 12 leads, 10s at 250Hz
    Output: (batch, embed_dim) — fixed-size feature vector

    MPS-optimised: kernel_size=7, max 256 channels (~5M params total).
    Effective receptive field at output covers the full 2500-sample signal.
    """

    def __init__(self, embed_dim: int = 256, dropout: float = 0.1):
        super().__init__()

        # Stem: project 12 leads into feature space
        self.stem = nn.Sequential(
            nn.Conv1d(12, 64, kernel_size=7, padding=3, bias=False),
            nn.BatchNorm1d(64),
            nn.ReLU(inplace=True),
        )

        # Stage 1: 64ch, length 2500
        self.stage1 = nn.Sequential(
            ResBlock1D(64, dropout=dropout),
            ResBlock1D(64, dropout=dropout),
        )

        # Stage 2: 128ch, length 1250 (stride 2)
        self.stage2 = nn.Sequential(
            DownsampleBlock1D(64, 128, stride=2),
            ResBlock1D(128, dropout=dropout),
            ResBlock1D(128, dropout=dropout),
        )

        # Stage 3: 256ch, length 625 (stride 2)
        self.stage3 = nn.Sequential(
            DownsampleBlock1D(128, 256, stride=2),
            ResBlock1D(256, dropout=dropout),
            ResBlock1D(256, dropout=dropout),
        )

        # Stage 4: 256ch, length 313 (stride 2) — keep 256 to stay MPS-memory-safe
        self.stage4 = nn.Sequential(
            DownsampleBlock1D(256, 256, stride=2),
            ResBlock1D(256, dropout=dropout),
        )

        self.pool = nn.AdaptiveAvgPool1d(1)
        self.embed_dim = 256

    def forward(self, x):
        # x: (batch, 12, 2500)
        x = self.stem(x)
        x = self.stage1(x)
        x = self.stage2(x)
        x = self.stage3(x)
        x = self.stage4(x)
        x = self.pool(x)          # (batch, 512, 1)
        return x.squeeze(-1)      # (batch, 512)


class TabularEncoder(nn.Module):
    """
    MLP encoder for ECG-derived tabular features.

    Input features (7): sex, ventricular_rate, atrial_rate, pr_interval,
                        qrs_duration, qt_corrected, age_at_ecg
    Output: (batch, 128) feature vector
    """

    def __init__(self, in_features: int = 7, dropout: float = 0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_features, 64),
            nn.LayerNorm(64),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(64, 128),
            nn.LayerNorm(128),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.net(x)


class EchoNextModel(nn.Module):
    """
    Multimodal 1D ResNet model for structural heart disease detection.

    Fuses ECG waveform features with tabular clinical features to predict
    12 binary SHD conditions simultaneously.

    Inputs:
        waveform: (batch, 12, 2500)  float32
        tabular:  (batch, 7)         float32  (pass zeros if unavailable)

    Outputs:
        logits:   (batch, 12)        float32  (apply sigmoid for probabilities)

    Label order (matches EchoNext metadata):
        0:  lvef_lte_45_flag
        1:  lvwt_gte_13_flag
        2:  aortic_stenosis_moderate_or_greater_flag
        3:  aortic_regurgitation_moderate_or_greater_flag
        4:  mitral_regurgitation_moderate_or_greater_flag
        5:  tricuspid_regurgitation_moderate_or_greater_flag
        6:  pulmonary_regurgitation_moderate_or_greater_flag
        7:  rv_systolic_dysfunction_moderate_or_greater_flag
        8:  pericardial_effusion_moderate_large_flag
        9:  pasp_gte_45_flag
        10: tr_max_gte_32_flag
        11: shd_moderate_or_greater_flag  (composite)
    """

    LABEL_NAMES = [
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

    LABEL_DISPLAY = [
        "Reduced EF (LVEF \u2264 45%)",
        "LV Wall Thickening (\u2265 1.3 cm)",
        "Aortic Stenosis (Mod+)",
        "Aortic Regurgitation (Mod+)",
        "Mitral Regurgitation (Mod+)",
        "Tricuspid Regurgitation (Mod+)",
        "Pulmonary Regurgitation (Mod+)",
        "RV Systolic Dysfunction (Mod+)",
        "Pericardial Effusion (Mod+)",
        "Pulmonary HTN (PASP \u2265 45 mmHg)",
        "Elevated TR Velocity (\u2265 3.2 m/s)",
        "Any Structural Heart Disease (Composite)",
    ]

    def __init__(self, dropout: float = 0.2):
        super().__init__()
        self.ecg_encoder = ECGEncoder1D(dropout=dropout)
        self.tab_encoder = TabularEncoder(dropout=dropout)

        # Fusion head: 256 (ecg) + 128 (tabular) = 384
        self.fusion = nn.Sequential(
            nn.Linear(256 + 128, 256),
            nn.LayerNorm(256),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(256, 64),
            nn.ReLU(inplace=True),
        )
        self.head = nn.Linear(64, 12)

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, waveform: torch.Tensor, tabular: torch.Tensor) -> torch.Tensor:
        ecg_feat = self.ecg_encoder(waveform)      # (B, 512)
        tab_feat = self.tab_encoder(tabular)        # (B, 128)
        fused = torch.cat([ecg_feat, tab_feat], dim=1)  # (B, 640)
        fused = self.fusion(fused)                  # (B, 64)
        return self.head(fused)                     # (B, 12)

    def predict_proba(self, waveform: torch.Tensor, tabular: torch.Tensor) -> torch.Tensor:
        """Return calibrated probabilities (0-1) for each condition."""
        with torch.no_grad():
            logits = self.forward(waveform, tabular)
            return torch.sigmoid(logits)
