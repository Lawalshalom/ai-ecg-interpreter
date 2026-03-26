"""
Inference engine for the EchoNext ECG interpreter.

Handles:
  - Model loading from checkpoint
  - Waveform preprocessing to match training distribution
  - Tabular feature preparation (from extracted values or user input)
  - Confidence-calibrated probability output
  - Structured result formatting for display
"""

import os
import json
import sys
from typing import Optional

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from model.architecture import EchoNextModel
from model.ptbxl_model import PTBXLModel
from app.diagnosis import DiagnosisEngine

_diagnosis_engine = DiagnosisEngine()


class PTBXLInferenceEngine:
    """
    Lightweight wrapper that runs the PTB-XL ECG diagnostic classifier.
    Returns None gracefully when the checkpoint does not yet exist
    (i.e. before training completes).
    """

    def __init__(self, checkpoint_path: Optional[str] = None,
                 device: Optional[str] = None):
        self.device = self._get_device(device)
        self.model  = PTBXLModel().to(self.device)
        self.model.eval()
        self.checkpoint_loaded = False
        self.val_auroc: Optional[float] = None

        if checkpoint_path is None:
            _ckpt_dir = os.path.join(
                os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                "checkpoints",
            )
            checkpoint_path = os.path.join(_ckpt_dir, "ptbxl_best_model.pt")

        if os.path.exists(checkpoint_path):
            ckpt = torch.load(
                checkpoint_path, map_location=self.device, weights_only=False
            )
            self.model.load_state_dict(ckpt["model_state_dict"])
            self.val_auroc = ckpt.get("val_mean_auroc")
            self.checkpoint_loaded = True
            print(f"[PTB-XL] Loaded checkpoint: {checkpoint_path}")
            if self.val_auroc:
                print(f"[PTB-XL] Val AUROC: {self.val_auroc:.4f}")
        else:
            print(f"[PTB-XL] No checkpoint at {checkpoint_path} — "
                  "run training/ptbxl_train.py to train.")

    def _get_device(self, device: Optional[str]) -> torch.device:
        if device:
            return torch.device(device)
        if torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")

    def predict(self, waveform: np.ndarray) -> Optional[dict]:
        """Returns probability dict or None if not trained."""
        if not self.checkpoint_loaded:
            return None
        w = waveform.squeeze()
        if w.shape == (2500, 12):
            w = w.T  # -> (12, 2500)

        # Detect 4× tiling artifact: digitizer tiles a 2.5 s crop 4× to fill 10 s.
        # Correlation ≈ 1.0 between [0:625] and [625:1250] for all tiled waveforms.
        # Fix: zero-pad beyond sample 625 (with cosine taper to avoid hard edge)
        # so the model does not interpret the repeat boundary as MI.
        first  = w[:, :625]
        second = w[:, 625:1250]
        flat_f = first.flatten()
        flat_s = second.flatten()
        if flat_f.std() > 1e-6 and flat_s.std() > 1e-6:
            corr = float(np.corrcoef(flat_f, flat_s)[0, 1])
            is_tiled = corr > 0.99
        else:
            is_tiled = False

        if is_tiled:
            taper_len = 50
            w_ptbxl = np.zeros((12, 2500), dtype=np.float32)
            w_ptbxl[:, :625 - taper_len] = w[:, :625 - taper_len]
            taper = np.cos(np.linspace(0, np.pi / 2, taper_len)).astype(np.float32)
            w_ptbxl[:, 625 - taper_len:625] = w[:, 625 - taper_len:625] * taper[None, :]
        else:
            w_ptbxl = w

        wave_t = torch.tensor(w_ptbxl, dtype=torch.float32).unsqueeze(0).to(self.device)
        with torch.no_grad():
            probs = torch.sigmoid(self.model(wave_t)).squeeze(0).cpu().numpy()
        return {
            "probabilities": [float(p) for p in probs],
            "labels":        PTBXLModel.LABEL_DISPLAY,
            "label_keys":    PTBXLModel.LABEL_NAMES,
        }


_ptbxl_engine = PTBXLInferenceEngine()

CHECKPOINT_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "checkpoints")

# EchoNext tabular feature column order
TABULAR_COLS = [
    "sex", "ventricular_rate", "atrial_rate", "pr_interval",
    "qrs_duration", "qt_corrected", "age_at_ecg",
]

# Fallback medians from EchoNext dataset (used when feature unavailable)
TABULAR_MEDIANS = {
    "sex": 1.0,
    "ventricular_rate": 79.0,
    "atrial_rate": 79.0,
    "pr_interval": 160.0,
    "qrs_duration": 90.0,
    "qt_corrected": 450.0,
    "age_at_ecg": 65.0,
}

# EchoNext training set stats for tabular standardization
# (computed from echonext_metadata_100k.csv)
TABULAR_MEANS = [0.57, 79.4, 72.8, 158.2, 93.1, 451.3, 63.8]
TABULAR_STDS  = [0.50, 19.8, 23.6,  32.4,  22.8,  37.6, 16.2]


class ECGInferenceEngine:
    """
    Loads the trained EchoNextModel and runs inference on processed ECG data.
    """

    def __init__(self, checkpoint_path: Optional[str] = None, device: Optional[str] = None):
        self.device = self._get_device(device)
        self.model = EchoNextModel().to(self.device)
        self.model.eval()

        # Load normalization stats if available
        self.norm_stats = self._load_norm_stats()

        # Load checkpoint
        if checkpoint_path is None:
            checkpoint_path = os.path.join(CHECKPOINT_DIR, "best_model.pt")

        self.checkpoint_loaded = False
        if os.path.exists(checkpoint_path):
            self._load_checkpoint(checkpoint_path)
        else:
            print(f"[Inference] No checkpoint found at {checkpoint_path}. Using random weights (for testing only).")

    def _get_device(self, device: Optional[str]) -> torch.device:
        if device:
            return torch.device(device)
        if torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")

    def _load_checkpoint(self, path: str):
        ckpt = torch.load(path, map_location=self.device, weights_only=True)
        self.model.load_state_dict(ckpt["model_state_dict"])
        self.val_auroc = ckpt.get("val_mean_auroc", None)
        self.checkpoint_loaded = True
        print(f"[Inference] Loaded checkpoint: {path}")
        if self.val_auroc:
            print(f"[Inference] Validation AUROC at checkpoint: {self.val_auroc:.4f}")

    def _load_norm_stats(self) -> Optional[dict]:
        path = os.path.join(CHECKPOINT_DIR, "norm_stats.json")
        if os.path.exists(path):
            with open(path) as f:
                return json.load(f)
        return None

    def predict(
        self,
        waveform: np.ndarray,
        patient_info: Optional[dict] = None,
    ) -> dict:
        """
        Run inference on a digitized ECG waveform.

        Args:
            waveform:     np.ndarray, shape (2500, 12) or (1, 2500, 12), float32
            patient_info: dict with any of: age, sex (0=female/1=male),
                         ventricular_rate, atrial_rate, pr_interval,
                         qrs_duration, qt_corrected

        Returns:
            result dict with keys:
              'probabilities'  -> list[float], 12 values 0-1
              'labels'         -> list[str], display names
              'shd_risk'       -> float, composite SHD probability
              'risk_level'     -> str, 'LOW' / 'MODERATE' / 'HIGH'
              'top_findings'   -> list[dict], conditions above threshold
              'ecg_features'   -> dict, extracted ECG features
              'model_loaded'   -> bool
        """
        # Prepare waveform tensor: (1, 12, 2500)
        wave = self._prepare_waveform(waveform)

        # Prepare tabular tensor: (1, 7)
        tab, ecg_features = self._prepare_tabular(waveform, patient_info)

        # Morphological ECG analysis (ST/Q waves — model-independent)
        ecg_morph = self._extract_morphological_features(waveform)

        # EchoNext structural inference
        wave_t = torch.tensor(wave, dtype=torch.float32).unsqueeze(0).to(self.device)
        tab_t  = torch.tensor(tab, dtype=torch.float32).unsqueeze(0).to(self.device)

        with torch.no_grad():
            logits = self.model(wave_t, tab_t)
            probs  = torch.sigmoid(logits).squeeze(0).cpu().numpy()

        # PTB-XL ECG diagnostic inference (runs only when checkpoint exists)
        ptbxl_result = _ptbxl_engine.predict(waveform)

        # Format results
        return self._format_results(probs, ecg_features, ecg_morph, ptbxl_result)

    def _prepare_waveform(self, waveform: np.ndarray) -> np.ndarray:
        """Ensure waveform is shape (12, 2500)."""
        w = waveform.squeeze()
        if w.shape == (2500, 12):
            return w.T.astype(np.float32)   # -> (12, 2500)
        if w.shape == (12, 2500):
            return w.astype(np.float32)
        raise ValueError(f"Unexpected waveform shape: {waveform.shape}. Expected (2500,12) or (12,2500).")

    def _prepare_tabular(self, waveform: np.ndarray, patient_info: Optional[dict]) -> tuple[np.ndarray, dict]:
        """
        Build standardized tabular feature vector.
        Extracts ECG features from waveform where possible,
        uses patient_info for demographics, falls back to medians.
        """
        info = patient_info or {}

        # Extract basic ECG features from digitized waveform
        ecg_features = self._extract_ecg_features(waveform)

        # Merge: patient_info > extracted features > medians
        raw_features = {}
        for col in TABULAR_COLS:
            if col in info and info[col] is not None:
                raw_features[col] = float(info[col])
            elif col in ecg_features and ecg_features[col] is not None:
                raw_features[col] = float(ecg_features[col])
            else:
                raw_features[col] = TABULAR_MEDIANS[col]

        # Standardize using training stats
        raw_arr = np.array([raw_features[c] for c in TABULAR_COLS], dtype=np.float32)
        means = np.array(TABULAR_MEANS, dtype=np.float32)
        stds  = np.array(TABULAR_STDS, dtype=np.float32)
        standardized = (raw_arr - means) / np.maximum(stds, 1e-6)

        ecg_features["_raw_features"] = raw_features
        return standardized, ecg_features

    def _extract_ecg_features(self, waveform: np.ndarray) -> dict:
        """
        Extract basic ECG features from the digitized waveform using NeuroKit2.
        Falls back gracefully if neurokit2 is unavailable.
        """
        features = {
            "ventricular_rate": None,
            "atrial_rate": None,
            "pr_interval": None,
            "qrs_duration": None,
            "qt_corrected": None,
        }
        try:
            import neurokit2 as nk

            w = waveform.squeeze()
            if w.ndim == 1:
                lead_ii = w
            elif w.shape == (2500, 12):
                lead_ii = w[:, 1]   # Lead II
            elif w.shape == (12, 2500):
                lead_ii = w[1, :]   # Lead II
            else:
                return features

            # Clean and process Lead II
            cleaned = nk.ecg_clean(lead_ii, sampling_rate=250)
            _, rpeaks = nk.ecg_peaks(cleaned, sampling_rate=250)
            rr_intervals = np.diff(rpeaks["ECG_R_Peaks"]) / 250.0  # seconds

            if len(rr_intervals) > 1:
                mean_rr = np.mean(rr_intervals)
                hr = 60.0 / mean_rr if mean_rr > 0 else None
                features["ventricular_rate"] = round(hr, 1) if hr else None
                features["atrial_rate"] = features["ventricular_rate"]

            # QRS and intervals via delineation
            try:
                _, waves = nk.ecg_delineate(cleaned, rpeaks, sampling_rate=250, method="dwt")
                q_peaks = np.array([x for x in waves["ECG_Q_Peaks"] if not np.isnan(x)])
                s_peaks = np.array([x for x in waves["ECG_S_Peaks"] if not np.isnan(x)])
                t_off   = np.array([x for x in waves["ECG_T_Offsets"] if not np.isnan(x)])
                p_on    = np.array([x for x in waves["ECG_P_Onsets"] if not np.isnan(x)])
                r_peaks_arr = rpeaks["ECG_R_Peaks"]

                if len(q_peaks) > 0 and len(s_peaks) > 0:
                    qrs_dur = np.median(s_peaks - q_peaks) / 250.0 * 1000  # ms
                    features["qrs_duration"] = round(float(qrs_dur), 1)

                if len(p_on) > 0 and len(r_peaks_arr) > 0:
                    min_len = min(len(p_on), len(r_peaks_arr))
                    pr = np.median(r_peaks_arr[:min_len] - p_on[:min_len]) / 250.0 * 1000  # ms
                    features["pr_interval"] = round(float(pr), 1)

                if len(t_off) > 0 and len(q_peaks) > 0 and features["ventricular_rate"]:
                    min_len = min(len(t_off), len(q_peaks))
                    qt = np.median(t_off[:min_len] - q_peaks[:min_len]) / 250.0 * 1000  # ms
                    # Bazett correction: QTc = QT / sqrt(RR in seconds)
                    rr_s = 60.0 / features["ventricular_rate"]
                    qtc = qt / np.sqrt(rr_s) if rr_s > 0 else qt
                    features["qt_corrected"] = round(float(qtc), 1)
            except Exception:
                pass  # Delineation can fail on noisy signals

        except Exception:
            pass  # NeuroKit2 unavailable or signal too noisy

        return features

    def _extract_morphological_features(self, waveform: np.ndarray) -> dict:
        """
        Detect ST elevation/depression and pathological Q waves per lead group.

        Lead order (EchoNext / digitizer standard):
            0=I  1=II  2=III  3=aVR  4=aVL  5=aVF  6=V1  7=V2  8=V3  9=V4  10=V5  11=V6

        The waveform is per-lead z-score normalised (mean=0, std=1).
        ST deviations and Q depths are expressed in those normalised units.
        Empirical threshold: 0.20 normalised units ≈ 1 mm (0.1 mV) on a typical ECG.
        """
        morph = {
            "st_elevation_inferior":  None,   # leads II, III, aVF
            "st_elevation_anterior":  None,   # leads V1-V4
            "st_elevation_lateral":   None,   # leads I, aVL, V5, V6
            "st_depression_inferior": None,
            "st_depression_anterior": None,
            "pathological_q_inferior": False,
            "pathological_q_anterior": False,
            "ecg_mi_patterns": [],            # list of territory strings
            "irregular_rhythm": False,        # R-R CV > 0.15 → AF/arrhythmia
            "rr_cv": None,                    # coefficient of variation of R-R
            "wide_qrs": False,                # QRS duration > 115 ms → BBB
            "qrs_duration_ms": None,          # estimated QRS duration in ms
        }
        try:
            import neurokit2 as nk

            w = waveform.squeeze()
            if w.shape == (2500, 12):
                arr = w
            elif w.shape == (12, 2500):
                arr = w.T
            else:
                return morph

            INFERIOR = [1, 2, 5]        # II, III, aVF
            ANTERIOR = [6, 7, 8, 9]     # V1–V4
            LATERAL  = [0, 4, 10, 11]   # I, aVL, V5, V6

            def _measure_lead(sig):
                """Return (median_ST_level, list_of_Q_info) in normalised units."""
                try:
                    cleaned = nk.ecg_clean(sig, sampling_rate=250)
                    _, rpeaks = nk.ecg_peaks(cleaned, sampling_rate=250)
                    r_pos = rpeaks["ECG_R_Peaks"]
                    if len(r_pos) < 2:
                        return None, []
                    _, waves = nk.ecg_delineate(
                        cleaned, rpeaks, sampling_rate=250, method="dwt"
                    )
                    s_peaks = [int(x) for x in waves.get("ECG_S_Peaks", [])
                               if not np.isnan(x)]
                    q_peaks = [int(x) for x in waves.get("ECG_Q_Peaks", [])
                               if not np.isnan(x)]

                    st_vals, q_info = [], []
                    for r_idx in r_pos:
                        # Isoelectric baseline: 100 ms window ending 40 ms before R
                        bl_end   = max(0, r_idx - 10)
                        bl_start = max(0, bl_end - 25)
                        baseline = float(np.mean(cleaned[bl_start:bl_end])) if bl_end > bl_start else 0.0

                        # J point: 40 ms after nearest S wave, else 60 ms after R
                        s_near = [s for s in s_peaks if 0 < s - r_idx < 50]
                        j0 = (s_near[0] + 10) if s_near else (r_idx + 15)
                        j1 = min(j0 + 20, len(cleaned) - 1)
                        if j1 > j0:
                            st_vals.append(float(np.mean(cleaned[j0:j1])) - baseline)

                        # R amplitude (for ratio thresholding)
                        r_amp = float(cleaned[r_idx]) - baseline if r_idx < len(cleaned) else 1.0

                        # Q wave: nearest Q peak within 30 samples before R
                        q_near = [q for q in q_peaks if -30 < q - r_idx < 0]
                        if q_near and r_amp > 0.1:
                            q_idx   = q_near[0]
                            q_depth = abs(float(cleaned[q_idx]) - baseline)
                            q_width = r_idx - q_idx
                            q_info.append({
                                "depth_ratio":   q_depth / r_amp,
                                "width_samples": q_width,
                            })

                    return (float(np.median(st_vals)) if st_vals else None), q_info
                except Exception:
                    return None, []

            # Concordant ST elevation: require ≥2 leads individually > 0.65
            # Higher threshold prevents Gaussian T-wave upslope overlap from
            # triggering false positives in synthetic/clean ECGs (≈0.60–0.62 range).
            # Real acute MI ST elevation (e.g. 0.85–1.2 units) is well above this.
            PER_LEAD_ST_THRESHOLD = 0.65
            MIN_CONCORDANT_LEADS  = 2

            for group, indices, name in [
                (INFERIOR, INFERIOR, "inferior"),
                (ANTERIOR, ANTERIOR, "anterior"),
                (LATERAL,  LATERAL,  "lateral"),
            ]:
                st_per_lead, q_patho, total = [], 0, 0
                for idx in indices:
                    st, q_list = _measure_lead(arr[:, idx])
                    if st is not None:
                        st_per_lead.append(st)
                    # Pathological Q: depth > 25% R AND width >= 8 samples (32 ms)
                    patho = any(
                        qi["depth_ratio"] > 0.25 and qi["width_samples"] >= 8
                        for qi in q_list
                    )
                    if patho:
                        q_patho += 1
                    total += 1

                if st_per_lead:
                    qualifying = [s for s in st_per_lead if s > PER_LEAD_ST_THRESHOLD]
                    if len(qualifying) >= MIN_CONCORDANT_LEADS:
                        morph[f"st_elevation_{name}"] = float(np.mean(qualifying))
                    else:
                        # Still capture depression (negative mean)
                        mean_st = float(np.mean(st_per_lead))
                        if mean_st < 0:
                            morph[f"st_depression_{name}"] = abs(mean_st)

                # Pathological Q in ≥2 leads of the group
                if total > 0 and q_patho >= min(2, total):
                    morph[f"pathological_q_{name}"] = True

            # Determine MI territories (only when concordant ST elevation confirmed)
            patterns = []
            for name in ("inferior", "anterior", "lateral"):
                st_elev = morph.get(f"st_elevation_{name}") or 0.0
                has_q   = morph.get(f"pathological_q_{name}", False)
                if st_elev > 0 or has_q:
                    patterns.append(name)
            morph["ecg_mi_patterns"] = patterns

            # ── Rhythm analysis: R-R coefficient of variation for AF detection ──
            # Use first 1500 samples (6 s) of Lead II to capture multiple beats.
            # Tiling makes samples 625–1249 identical to 0–624, but individual
            # beats within that range are real and usable for QRS-width detection.
            try:
                lead_ii_full = arr[:1500, 1]  # first 6 s from Lead II
                cleaned_ii = nk.ecg_clean(lead_ii_full, sampling_rate=250)
                _, rpeaks_ii = nk.ecg_peaks(cleaned_ii, sampling_rate=250)
                rr_ii = np.diff(rpeaks_ii["ECG_R_Peaks"]) / 250.0  # seconds
                if len(rr_ii) >= 4:
                    rr_cv = float(np.std(rr_ii) / np.mean(rr_ii)) if np.mean(rr_ii) > 0 else 0.0
                    morph["rr_cv"] = round(rr_cv, 3)
                    morph["irregular_rhythm"] = rr_cv > 0.15
            except Exception:
                pass

            # ── QRS width analysis for bundle branch block detection ──
            # Use Lead I (index 0) — clear QRS in both LBBB and RBBB.
            try:
                lead_i = arr[:625, 0]   # single real cycle window
                cleaned_i = nk.ecg_clean(lead_i, sampling_rate=250)
                _, rp_i = nk.ecg_peaks(cleaned_i, sampling_rate=250)
                if len(rp_i["ECG_R_Peaks"]) >= 1:
                    _, waves_i = nk.ecg_delineate(cleaned_i, rp_i, sampling_rate=250, method="dwt")
                    q_pts = [x for x in waves_i.get("ECG_Q_Peaks", []) if not np.isnan(x)]
                    s_pts = [x for x in waves_i.get("ECG_S_Peaks", []) if not np.isnan(x)]
                    if q_pts and s_pts:
                        qrs_samples = float(np.median([
                            s - q for q, s in zip(
                                sorted(q_pts), sorted(s_pts)
                            ) if s > q
                        ]))
                        qrs_ms = qrs_samples / 250.0 * 1000
                        morph["qrs_duration_ms"] = round(qrs_ms, 1)
                        morph["wide_qrs"] = qrs_ms > 115  # >115 ms → BBB
            except Exception:
                pass

        except Exception:
            pass

        return morph

    def _format_results(self, probs: np.ndarray, ecg_features: dict,
                        ecg_morph: Optional[dict] = None,
                        ptbxl_result: Optional[dict] = None) -> dict:
        """Structure inference output for Streamlit display."""
        labels = EchoNextModel.LABEL_DISPLAY
        label_keys = EchoNextModel.LABEL_NAMES

        shd_risk = float(probs[11])  # EchoNext Composite SHD (raw)

        # ── Recalibrate risk using ECG morphological features ──────────────────
        # The EchoNext model was trained on echocardiogram data; its raw SHD
        # output over-estimates risk for normal ECGs when fed ECG waveforms.
        # Morphological analysis provides direct, reliable discriminators:
        #   • Concordant ST elevation (≥2 leads > 0.50) → confirmed pathology
        #   • Irregular R-R (CV > 0.15)                → arrhythmia present
        # When neither is present, dampen EchoNext SHD by 50% to reduce FP.
        has_concordant_st = ecg_morph is not None and any(
            ecg_morph.get(f"st_elevation_{t}") for t in ("inferior", "anterior", "lateral")
        )
        is_irregular = ecg_morph is not None and ecg_morph.get("irregular_rhythm", False)

        if not has_concordant_st and not is_irregular:
            # No confirmed ECG pathology — EchoNext structural model is unreliable
            # for ECG inputs (trained on echo data). Dampen all 12 probabilities
            # by 50% to suppress false-positive structural diagnoses.
            probs = probs * 0.50
            shd_risk = float(probs[11])

        if shd_risk >= 0.60:
            risk_level = "HIGH"
        elif shd_risk >= 0.35:
            risk_level = "MODERATE"
        else:
            risk_level = "LOW"

        # Top findings: conditions >30% probability (excluding composite)
        top_findings = []
        for i in range(11):
            if probs[i] >= 0.30:
                top_findings.append({
                    "condition": labels[i],
                    "probability": float(probs[i]),
                    "flag": label_keys[i],
                })
        top_findings.sort(key=lambda x: x["probability"], reverse=True)

        raw_features = ecg_features.pop("_raw_features", {})

        # Normal ECG probability
        # P(normal) = (1 - composite_SHD) × (1 - worst_individual_finding)
        # When all findings are low this approaches 1; degrades quickly if any
        # condition is elevated.
        max_individual = float(np.max(probs[:11]))
        normal_ecg_prob = float((1.0 - shd_risk) * (1.0 - max_individual))

        # Clinical diagnosis layer (EchoNext + PTB-XL + ECG morphology)
        ptbxl_probs = ptbxl_result["probabilities"] if ptbxl_result else None
        diagnoses = _diagnosis_engine.diagnose(
            [float(p) for p in probs], ecg_morph or {}, ptbxl_probs
        )

        return {
            "probabilities":   [float(p) for p in probs],
            "labels":          labels,
            "label_keys":      label_keys,
            "shd_risk":        shd_risk,
            "risk_level":      risk_level,
            "top_findings":    top_findings,
            "ecg_features":    ecg_features,
            "raw_features":    raw_features,
            "normal_ecg_prob":    normal_ecg_prob,
            "diagnoses":          diagnoses,
            "ecg_morph":          ecg_morph or {},
            "ptbxl_result":       ptbxl_result,
            "ptbxl_model_loaded": _ptbxl_engine.checkpoint_loaded,
            "model_loaded":       self.checkpoint_loaded,
        }
