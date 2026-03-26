"""
PTB-XL ECG Diagnostic Classifier

12-class binary model trained on the PTB-XL v1.0.3 dataset.
Reuses the same ECGEncoder1D backbone as EchoNextModel.

Label order:
    0:  normal_ecg                — No significant ECG abnormality
    1:  inferior_mi               — Inferior wall MI (IMI, ILMI, IPMI, IPLMI)
    2:  anterior_mi               — Anterior MI (AMI, ASMI, ALMI)
    3:  lateral_mi                — Lateral MI (LMI)
    4:  posterior_mi              — Posterior MI (PMI)
    5:  stt_changes               — ST/T changes (ischaemia / non-specific)
    6:  lbbb                      — Left bundle branch block
    7:  rbbb                      — Right bundle branch block
    8:  av_block                  — AV conduction block (any degree)
    9:  lvh                       — Left ventricular hypertrophy
    10: atrial_fibrillation       — Atrial fibrillation or flutter
    11: any_mi_composite          — Any MI territory (composite)
"""

import os
import sys

import torch
import torch.nn as nn

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from model.architecture import ECGEncoder1D


class PTBXLModel(nn.Module):
    """
    12-class ECG pattern classifier trained on PTB-XL.

    Input:  waveform  (batch, 12, 2500)  float32
    Output: logits    (batch, 12)        float32  (apply sigmoid for probabilities)
    """

    LABEL_NAMES = [
        "normal_ecg",
        "inferior_mi",
        "anterior_mi",
        "lateral_mi",
        "posterior_mi",
        "stt_changes",
        "lbbb",
        "rbbb",
        "av_block",
        "lvh",
        "atrial_fibrillation",
        "any_mi_composite",
    ]

    LABEL_DISPLAY = [
        "Normal ECG",
        "Inferior Wall MI / Ischaemia",
        "Anterior MI / Ischaemia",
        "Lateral MI",
        "Posterior MI",
        "ST / T-wave Changes (Ischaemia)",
        "Left Bundle Branch Block (LBBB)",
        "Right Bundle Branch Block (RBBB)",
        "AV Conduction Block",
        "LV Hypertrophy",
        "Atrial Fibrillation / Flutter",
        "Any Myocardial Infarction (Composite)",
    ]

    def __init__(self, dropout: float = 0.2):
        super().__init__()
        self.ecg_encoder = ECGEncoder1D(dropout=dropout)

        # Classification head: 256 → 128 → 12
        self.classifier = nn.Sequential(
            nn.Linear(256, 128),
            nn.LayerNorm(128),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(128, 12),
        )
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

    def forward(self, waveform: torch.Tensor) -> torch.Tensor:
        feat = self.ecg_encoder(waveform)   # (B, 256)
        return self.classifier(feat)         # (B, 12)

    def predict_proba(self, waveform: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            return torch.sigmoid(self.forward(waveform))
