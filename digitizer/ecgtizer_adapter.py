"""
Adapter wrapping UMMISCO/ecgtizer for use in the ECG interpreter pipeline.

Converts ecgtizer output (dict of 5000-sample leads at ~500 Hz) to
(2500, 12) float32 at 250 Hz matching the EchoNext model input format.

Usage:
    from digitizer.ecgtizer_adapter import ECGtizerAdapter
    adapter = ECGtizerAdapter()
    waveform, meta = adapter.process("ecg.pdf")
    # waveform: np.ndarray (2500, 12) float32
"""

import os
import io
import warnings
import tempfile
import numpy as np
import scipy.signal as sp_signal

warnings.filterwarnings("ignore")

# EchoNext lead order (indices 0-11)
ECHONEXT_LEAD_ORDER = ['I', 'II', 'III', 'AVR', 'AVL', 'AVF', 'V1', 'V2', 'V3', 'V4', 'V5', 'V6']

TARGET_SAMPLES = 2500   # 10 s × 250 Hz


class ECGtizerAdapter:
    """
    Thin adapter around UMMISCO/ecgtizer that produces (2500, 12) float32 arrays.
    """

    def process(self, source) -> tuple[np.ndarray, dict]:
        meta = {"success": False, "notes": [], "source": "ecgtizer"}
        try:
            pdf_path, tmp = self._ensure_pdf(source, meta)
            leads_raw = self._run_ecgtizer(pdf_path, meta)
            if tmp:
                os.unlink(pdf_path)
            waveform = self._assemble_waveform(leads_raw, meta)
            waveform = self._normalize_echonext(waveform)
            meta["success"] = True
            meta["fs"] = 250
            return waveform, meta
        except Exception as e:
            import traceback
            meta["notes"].append(f"ERROR: {e}")
            meta["error"] = str(e)
            meta["traceback"] = traceback.format_exc()
            return np.zeros((TARGET_SAMPLES, 12), dtype=np.float32), meta

    # ── Helpers ────────────────────────────────────────────────────────────────

    def _ensure_pdf(self, source, meta: dict) -> tuple[str, bool]:
        """Return (pdf_path, is_temp). Converts image→PDF if needed."""
        from PIL import Image

        if isinstance(source, str):
            path = source
            if path.lower().endswith('.pdf'):
                return path, False
            # Image file → convert to PDF
            img = Image.open(path).convert('RGB')
            tmp = tempfile.NamedTemporaryFile(suffix='.pdf', delete=False)
            img.save(tmp.name, 'PDF', resolution=200)
            meta["notes"].append(f"Converted {os.path.basename(path)} → PDF")
            return tmp.name, True

        if isinstance(source, (bytes, io.BytesIO)):
            data = source if isinstance(source, bytes) else source.read()
            if data[:4] == b'%PDF':
                tmp = tempfile.NamedTemporaryFile(suffix='.pdf', delete=False)
                tmp.write(data); tmp.flush()
                return tmp.name, True
            # Image bytes → PIL → PDF
            pil = Image.open(io.BytesIO(data)).convert('RGB')
            tmp = tempfile.NamedTemporaryFile(suffix='.pdf', delete=False)
            pil.save(tmp.name, 'PDF', resolution=200)
            meta["notes"].append("Converted image bytes → PDF")
            return tmp.name, True

        raise ValueError(f"Unsupported source type: {type(source)}")

    def _run_ecgtizer(self, pdf_path: str, meta: dict) -> dict:
        """Run ecgtizer and return the extracted_lead dict."""
        # Ensure poppler is on PATH (macOS homebrew)
        if '/opt/homebrew/bin' not in os.environ.get('PATH', ''):
            os.environ['PATH'] = os.environ.get('PATH', '') + ':/opt/homebrew/bin'

        from ecgtizer import ECGtizer
        ecg = ECGtizer(
            file=pdf_path,
            dpi=300,
            extraction_method='full',
            verbose=False,
            DEBUG=False,
        )
        leads = ecg.extracted_lead or {}
        n_detected = sum(1 for v in leads.values()
                         if hasattr(v, '__len__') and np.array(v).std() > 0)
        meta["notes"].append(f"ecgtizer detected {n_detected}/12 leads with signal")
        meta["n_leads_detected"] = n_detected
        return leads

    def _assemble_waveform(self, leads_raw: dict, meta: dict) -> np.ndarray:
        """
        Map ecgtizer leads → (2500, 12) array in EchoNext lead order.
        Resamples from 5000 → 2500 samples.
        """
        result = np.zeros((TARGET_SAMPLES, 12), dtype=np.float32)
        notes = []

        for col_idx, lead_name in enumerate(ECHONEXT_LEAD_ORDER):
            raw = leads_raw.get(lead_name, None)
            if raw is None:
                notes.append(f"{lead_name}:missing")
                continue
            raw = np.array(raw, dtype=np.float64)
            if raw.std() < 1e-6:
                notes.append(f"{lead_name}:zero")
                continue

            # Resample from source length → 2500
            if len(raw) != TARGET_SAMPLES:
                resampled = sp_signal.resample(raw, TARGET_SAMPLES)
            else:
                resampled = raw

            result[:, col_idx] = resampled.astype(np.float32)
            notes.append(f"{lead_name}:ok")

        meta["notes"].append("Leads: " + ", ".join(notes))
        return result

    def _normalize_echonext(self, waveform: np.ndarray) -> np.ndarray:
        """Per-lead z-score normalization matching EchoNext training distribution."""
        out = waveform.copy()
        for i in range(out.shape[1]):
            lead = out[:, i]
            std = lead.std()
            if std > 1e-6:
                out[:, i] = np.clip((lead - lead.mean()) / std, -5.0, 5.0)
        return out.astype(np.float32)
