"""
Comprehensive TDD test suite for the ECG Interpreter.

Run with:
    cd /Users/mac/Downloads/ecg-interpreter
    pytest tests/test_ecg_interpreter.py -v

All tests must pass after every code change to prevent regressions.

Test matrix:
  - ECG validation: legitimate ECGs pass, non-ECGs fail
  - Digitizer output: correct shape, quality, normalization
  - Normal ECG (norm.png): no false positives — no MI, not HIGH risk
  - Inferior Wall MI: MI diagnosis with HIGH confidence
  - Anterior Wall MI: MI diagnosis in anterior territory
  - Atrial Fibrillation: AF / arrhythmia detected
  - LBBB: bundle-branch conduction abnormality detected
  - RBBB: bundle-branch conduction abnormality detected
  - VFib: life-threatening arrhythmia flagged HIGH risk
  - WPW: pre-excitation pattern detected
  - Edge cases: blank, portrait, solid-black images rejected
"""

import os
import sys
import io

import numpy as np
import pytest

# ── Path setup ────────────────────────────────────────────────────────────────
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

ECG_DIR   = "/Users/mac/Downloads/ECGs/"
NORM_PNG  = "/Users/mac/Downloads/norm.png"
RESUME    = "/Users/mac/Downloads/FemiLawalResume.pdf"


# ── Module-level fixtures (loaded once per session) ───────────────────────────

@pytest.fixture(scope="session")
def digitizer():
    from digitizer.pipeline import ECGDigitizer
    return ECGDigitizer()


@pytest.fixture(scope="session")
def engine():
    from app.inference import ECGInferenceEngine
    return ECGInferenceEngine()


def _analyze(digitizer, engine, path, age=55, sex=1):
    """Digitize + run inference. Returns (waveform, meta, result)."""
    waveform, meta = digitizer.process(path)
    result = engine.predict(waveform, {"age": age, "sex": sex})
    return waveform, meta, result


def _has_dx(result, *keywords):
    """Return True if any diagnosis name contains any of the keywords."""
    dxs = result.get("diagnoses", [])
    for dx in dxs:
        n = dx["name"].lower()
        if any(kw.lower() in n for kw in keywords):
            return True
    return False


def _top_dx_pct(result, *keywords):
    """Return highest confidence % for any diagnosis matching a keyword."""
    best = 0
    for dx in result.get("diagnoses", []):
        n = dx["name"].lower()
        if any(kw.lower() in n for kw in keywords):
            best = max(best, dx["pct"])
    return best


# ═══════════════════════════════════════════════════════════════════════════════
# 1. ECG VALIDATION
# ═══════════════════════════════════════════════════════════════════════════════

class TestECGValidation:
    """The digitizer must accept ECG images and reject non-ECG files."""

    def test_resume_pdf_rejected(self, digitizer):
        """Resume PDF must fail with a human-readable portrait/document error."""
        _, meta = digitizer.process(RESUME)
        assert not meta["success"], "Resume PDF should be rejected"
        err = meta.get("error", "").lower()
        assert any(kw in err for kw in ("portrait", "document", "landscape", "ecg tracing")), \
            f"Resume should give clear rejection message, got: {meta.get('error')}"

    @pytest.mark.parametrize("filename", [
        "inf wall mi.tif",
        "atrial fib.tif",
        "left bundle branch block.tif",
        "rbbb.tif",
        "ant wall mi.tif",
        "vfib.tif",
        "wpw.tif",
        "torsades de pointes.tif",
        "a-flutter.tif",
        "first degree av.tif",
        "monomorphic v tach.tif",
        "pericarditis.tif",
    ])
    def test_clinical_ecg_accepted(self, digitizer, filename):
        """All clinical ECG images in the test set must pass validation."""
        path = os.path.join(ECG_DIR, filename)
        _, meta = digitizer.process(path)
        assert meta["success"], \
            f"{filename} should be accepted but got: {meta.get('error')}"

    def test_norm_png_accepted(self, digitizer):
        _, meta = digitizer.process(NORM_PNG)
        assert meta["success"], f"norm.png rejected: {meta.get('error')}"

    def test_blank_image_rejected(self, digitizer):
        """A white (blank) image must be rejected."""
        from PIL import Image
        blank = np.ones((400, 600, 3), dtype=np.uint8) * 255
        img = Image.fromarray(blank)
        buf = io.BytesIO()
        img.save(buf, format="PNG")
        _, meta = digitizer.process(io.BytesIO(buf.getvalue()))
        assert not meta["success"], "Blank image should be rejected"

    def test_solid_black_rejected(self, digitizer):
        """A solid-black image must be rejected."""
        from PIL import Image
        black = np.zeros((400, 600, 3), dtype=np.uint8)
        img = Image.fromarray(black)
        buf = io.BytesIO()
        img.save(buf, format="PNG")
        _, meta = digitizer.process(io.BytesIO(buf.getvalue()))
        assert not meta["success"], "Solid black image should be rejected"

    def test_tiny_image_rejected(self, digitizer):
        """A very small image (< 100×150 px) must be rejected."""
        from PIL import Image
        small = np.random.randint(0, 255, (50, 80, 3), dtype=np.uint8)
        img = Image.fromarray(small)
        buf = io.BytesIO()
        img.save(buf, format="PNG")
        _, meta = digitizer.process(io.BytesIO(buf.getvalue()))
        assert not meta["success"], "Tiny image should be rejected"


# ═══════════════════════════════════════════════════════════════════════════════
# 2. DIGITIZER OUTPUT QUALITY
# ═══════════════════════════════════════════════════════════════════════════════

class TestDigitizerOutput:
    """Verify the digitizer produces correct-format, usable waveforms."""

    @pytest.fixture(autouse=True)
    def setup(self, digitizer):
        self.w, self.meta = digitizer.process(os.path.join(ECG_DIR, "inf wall mi.tif"))
        assert self.meta["success"], "Test setup: inf wall mi.tif must succeed"

    def test_output_shape(self):
        assert self.w.shape == (2500, 12), \
            f"Expected (2500, 12), got {self.w.shape}"

    def test_12_leads_detected(self):
        n = self.meta.get("n_leads_detected", 0)
        assert n >= 10, f"Expected ≥ 10 leads, got {n}"

    def test_waveform_has_signal(self):
        assert np.abs(self.w).max() > 0.1, "Waveform appears flat/empty"

    def test_waveform_normalized(self):
        """After z-score normalization values must be within ±5.5."""
        assert np.abs(self.w).max() <= 5.5, \
            f"Waveform not normalized: max={np.abs(self.w).max():.2f}"

    def test_leads_not_identical(self):
        """All 12 leads should not be identical (would indicate extraction failure)."""
        unique_leads = set()
        for i in range(12):
            lead_hash = hash(self.w[:100, i].tobytes())
            unique_leads.add(lead_hash)
        assert len(unique_leads) >= 6, \
            f"Too many identical leads — extraction likely failed ({len(unique_leads)} unique)"

    @pytest.mark.parametrize("filename", [
        "atrial fib.tif", "left bundle branch block.tif", "rbbb.tif",
    ])
    def test_multiple_ecgs_produce_valid_output(self, digitizer, filename):
        w, meta = digitizer.process(os.path.join(ECG_DIR, filename))
        assert meta["success"]
        assert w.shape == (2500, 12)
        assert np.abs(w).max() > 0.1


# ═══════════════════════════════════════════════════════════════════════════════
# 3. NORMAL ECG — no false positives
# ═══════════════════════════════════════════════════════════════════════════════

class TestNormalECG:
    """
    norm.png is a normal 12-lead ECG sourced online.
    The interpreter must NOT produce HIGH risk or MI diagnoses.
    """

    @pytest.fixture(autouse=True)
    def setup(self, digitizer, engine):
        self.w, self.meta, self.result = _analyze(digitizer, engine, NORM_PNG, age=40, sex=1)
        assert self.meta["success"], "norm.png must digitize successfully"

    def test_not_high_risk(self):
        level = self.result["risk_level"]
        assert level in ("LOW", "MODERATE"), \
            f"Normal ECG incorrectly flagged as {level} risk"

    def test_shd_risk_below_60pct(self):
        shd = self.result["shd_risk"]
        assert shd < 0.60, \
            f"Normal ECG SHD risk too high: {shd*100:.0f}% (expected < 60%)"

    def test_no_high_confidence_mi(self):
        """No MI diagnosis should exceed 70% confidence on a normal ECG."""
        mi_pct = _top_dx_pct(self.result, "mi", "infarction", "ischaemia")
        assert mi_pct < 70, \
            f"Normal ECG falsely diagnosed with MI at {mi_pct}% confidence"

    def test_no_high_confidence_ptbxl_mi(self):
        """PTB-XL model should not give >70% MI confidence on a normal ECG."""
        ptbxl = self.result.get("ptbxl_result")
        if ptbxl is None:
            pytest.skip("PTB-XL model not loaded")
        from model.ptbxl_model import PTBXLModel
        probs = dict(zip(PTBXLModel.LABEL_NAMES, ptbxl["probabilities"]))
        lat_mi = probs.get("lateral_mi", 0)
        ant_mi = probs.get("anterior_mi", 0)
        assert lat_mi < 0.70, \
            f"PTB-XL reports lateral MI at {lat_mi*100:.0f}% on a normal ECG"
        assert ant_mi < 0.70, \
            f"PTB-XL reports anterior MI at {ant_mi*100:.0f}% on a normal ECG"

    def test_normal_or_mild_findings_only(self):
        """If there are diagnoses, they should all be POSSIBLE (not HIGH)."""
        high_confidence = [
            d for d in self.result.get("diagnoses", [])
            if d["level"] == "HIGH" and "mi" in d["name"].lower()
        ]
        assert len(high_confidence) == 0, \
            f"Normal ECG has HIGH-confidence MI diagnoses: {[(d['name'], d['pct']) for d in high_confidence]}"


# ═══════════════════════════════════════════════════════════════════════════════
# 4. INFERIOR WALL MI
# ═══════════════════════════════════════════════════════════════════════════════

class TestInferiorWallMI:
    """inf wall mi.tif: ECG with acute inferior wall myocardial infarction."""

    @pytest.fixture(autouse=True)
    def setup(self, digitizer, engine):
        self.w, _, self.result = _analyze(
            digitizer, engine, os.path.join(ECG_DIR, "inf wall mi.tif"), age=55, sex=1
        )

    def test_high_risk(self):
        assert self.result["risk_level"] == "HIGH", \
            f"Inferior Wall MI should be HIGH risk, got {self.result['risk_level']}"

    def test_mi_diagnosis_present(self):
        assert _has_dx(self.result, "mi", "infarction", "ischaemia"), \
            f"No MI diagnosis found. Diagnoses: {[d['name'] for d in self.result.get('diagnoses', [])]}"

    def test_mi_high_confidence(self):
        pct = _top_dx_pct(self.result, "mi", "infarction", "ischaemia")
        assert pct >= 70, \
            f"MI diagnosis confidence too low: {pct}% (need ≥ 70%)"

    def test_shd_risk_high(self):
        assert self.result["shd_risk"] >= 0.60, \
            f"SHD risk too low for MI ECG: {self.result['shd_risk']*100:.0f}%"


# ═══════════════════════════════════════════════════════════════════════════════
# 5. ANTERIOR WALL MI
# ═══════════════════════════════════════════════════════════════════════════════

class TestAnteriorWallMI:
    """ant wall mi.tif: ECG with anterior wall MI."""

    @pytest.fixture(autouse=True)
    def setup(self, digitizer, engine):
        self.w, _, self.result = _analyze(
            digitizer, engine, os.path.join(ECG_DIR, "ant wall mi.tif"), age=55, sex=1
        )

    def test_high_risk(self):
        assert self.result["risk_level"] == "HIGH"

    def test_anterior_mi_or_ischaemia_diagnosed(self):
        assert _has_dx(self.result, "anterior", "mi", "infarction", "ischaemia"), \
            f"Anterior MI not diagnosed. Got: {[d['name'] for d in self.result.get('diagnoses', [])]}"

    def test_high_confidence(self):
        pct = _top_dx_pct(self.result, "anterior", "mi", "infarction")
        assert pct >= 70, f"Anterior MI confidence too low: {pct}%"


# ═══════════════════════════════════════════════════════════════════════════════
# 6. ATRIAL FIBRILLATION
# ═══════════════════════════════════════════════════════════════════════════════

class TestAtrialFibrillation:
    """atrial fib.tif: ECG showing atrial fibrillation."""

    @pytest.fixture(autouse=True)
    def setup(self, digitizer, engine):
        self.w, _, self.result = _analyze(
            digitizer, engine, os.path.join(ECG_DIR, "atrial fib.tif"), age=65, sex=0
        )

    def test_moderate_or_high_risk(self):
        assert self.result["risk_level"] in ("MODERATE", "HIGH"), \
            f"AF ECG should be MODERATE or HIGH risk, got {self.result['risk_level']}"

    def test_af_or_arrhythmia_diagnosed(self):
        assert _has_dx(self.result, "fibrillation", "flutter", "arrhythmia", "irregular"), \
            f"AF ECG should diagnose AF/arrhythmia. Got: {[d['name'] for d in self.result.get('diagnoses', [])]}"

    def test_af_confidence_adequate(self):
        pct = _top_dx_pct(self.result, "fibrillation", "flutter", "arrhythmia")
        assert pct >= 50, f"AF diagnosis confidence too low: {pct}%"


# ═══════════════════════════════════════════════════════════════════════════════
# 7. LEFT BUNDLE BRANCH BLOCK
# ═══════════════════════════════════════════════════════════════════════════════

class TestLBBB:
    """left bundle branch block.tif: ECG with complete LBBB."""

    @pytest.fixture(autouse=True)
    def setup(self, digitizer, engine):
        self.w, _, self.result = _analyze(
            digitizer, engine,
            os.path.join(ECG_DIR, "left bundle branch block.tif"), age=60, sex=1
        )

    def test_conduction_abnormality_diagnosed(self):
        assert _has_dx(self.result, "bundle", "block", "conduction", "lbbb"), \
            f"LBBB ECG should show bundle branch block. Got: {[d['name'] for d in self.result.get('diagnoses', [])]}"

    def test_conduction_confidence_adequate(self):
        pct = _top_dx_pct(self.result, "bundle", "block", "conduction", "lbbb")
        assert pct >= 50, f"LBBB confidence too low: {pct}%"


# ═══════════════════════════════════════════════════════════════════════════════
# 8. RIGHT BUNDLE BRANCH BLOCK
# ═══════════════════════════════════════════════════════════════════════════════

class TestRBBB:
    """rbbb.tif: ECG with right bundle branch block."""

    @pytest.fixture(autouse=True)
    def setup(self, digitizer, engine):
        self.w, _, self.result = _analyze(
            digitizer, engine, os.path.join(ECG_DIR, "rbbb.tif"), age=50, sex=1
        )

    def test_conduction_abnormality_diagnosed(self):
        assert _has_dx(self.result, "bundle", "block", "conduction", "rbbb"), \
            f"RBBB ECG should show bundle branch block. Got: {[d['name'] for d in self.result.get('diagnoses', [])]}"

    def test_conduction_confidence_adequate(self):
        pct = _top_dx_pct(self.result, "bundle", "block", "conduction", "rbbb")
        assert pct >= 50, f"RBBB confidence too low: {pct}%"


# ═══════════════════════════════════════════════════════════════════════════════
# 9. VENTRICULAR FIBRILLATION (life-threatening)
# ═══════════════════════════════════════════════════════════════════════════════

class TestVFib:
    """vfib.tif: Ventricular fibrillation — must flag as HIGH risk."""

    @pytest.fixture(autouse=True)
    def setup(self, digitizer, engine):
        self.w, _, self.result = _analyze(
            digitizer, engine, os.path.join(ECG_DIR, "vfib.tif"), age=55, sex=1
        )

    def test_high_risk(self):
        assert self.result["risk_level"] == "HIGH", \
            f"VFib should be HIGH risk, got {self.result['risk_level']}"

    def test_high_shd_risk(self):
        assert self.result["shd_risk"] >= 0.50, \
            f"VFib SHD risk too low: {self.result['shd_risk']*100:.0f}%"


# ═══════════════════════════════════════════════════════════════════════════════
# 10. WPW SYNDROME
# ═══════════════════════════════════════════════════════════════════════════════

class TestWPW:
    """wpw.tif: Wolff-Parkinson-White syndrome with delta waves."""

    @pytest.fixture(autouse=True)
    def setup(self, digitizer, engine):
        self.w, _, self.result = _analyze(
            digitizer, engine, os.path.join(ECG_DIR, "wpw.tif"), age=30, sex=1
        )

    def test_not_low_risk(self):
        """WPW is a pre-excitation syndrome — should not be LOW risk."""
        assert self.result["risk_level"] in ("MODERATE", "HIGH"), \
            f"WPW should be at least MODERATE risk, got {self.result['risk_level']}"

    def test_has_some_diagnosis(self):
        assert len(self.result.get("diagnoses", [])) > 0, \
            "WPW ECG produced no diagnoses"


# ═══════════════════════════════════════════════════════════════════════════════
# 11. PERICARDITIS
# ═══════════════════════════════════════════════════════════════════════════════

class TestPericarditis:
    """pericarditis.tif: ECG changes from pericarditis (diffuse ST elevation)."""

    @pytest.fixture(autouse=True)
    def setup(self, digitizer, engine):
        self.w, _, self.result = _analyze(
            digitizer, engine, os.path.join(ECG_DIR, "pericarditis.tif"), age=35, sex=1
        )

    def test_pericardial_or_ischaemia_finding(self):
        assert _has_dx(self.result, "pericardial", "ischaemia", "st", "inflammation") \
               or self.result["shd_risk"] >= 0.35, \
            f"Pericarditis ECG missed. Got risk={self.result['risk_level']}, dxs={[d['name'] for d in self.result.get('diagnoses', [])]}"


# ═══════════════════════════════════════════════════════════════════════════════
# 12. EDGE CASES
# ═══════════════════════════════════════════════════════════════════════════════

class TestEdgeCases:
    """Edge cases the system must handle gracefully."""

    def test_predict_with_zeros_does_not_crash(self, engine):
        """Inference on an all-zero waveform must not raise an exception."""
        zeros = np.zeros((2500, 12), dtype=np.float32)
        result = engine.predict(zeros, {"age": 50, "sex": 1})
        assert "risk_level" in result
        assert "diagnoses" in result

    def test_predict_with_synthetic_normal_ecg(self, engine):
        """The demo (synthetic) normal ECG should not be HIGH risk."""
        from digitizer.pipeline import load_demo_ecg
        waveform, _ = load_demo_ecg()
        result = engine.predict(waveform, {"age": 40, "sex": 1})
        assert result["risk_level"] in ("LOW", "MODERATE"), \
            f"Synthetic normal ECG should not be HIGH risk, got {result['risk_level']}"

    def test_waveform_shape_12x2500_accepted(self, engine):
        """Engine must accept (12, 2500) as well as (2500, 12) input."""
        w = np.random.randn(12, 2500).astype(np.float32)
        result = engine.predict(w, {})
        assert "risk_level" in result

    def test_missing_patient_info_uses_defaults(self, engine):
        """Inference must succeed when patient_info is None or empty."""
        w = np.random.randn(2500, 12).astype(np.float32)
        for info in [None, {}, {"age": 45}]:
            result = engine.predict(w, info)
            assert "risk_level" in result, f"Failed with patient_info={info}"

    def test_all_ecgs_produce_valid_results(self, digitizer, engine):
        """Every ECG in the test library must produce a valid result end-to-end."""
        for filename in os.listdir(ECG_DIR):
            if not filename.endswith(".tif"):
                continue
            path = os.path.join(ECG_DIR, filename)
            w, meta = digitizer.process(path)
            if not meta["success"]:
                pytest.fail(f"{filename} failed digitization: {meta.get('error')}")
            result = engine.predict(w, {"age": 55, "sex": 1})
            assert result["risk_level"] in ("LOW", "MODERATE", "HIGH"), \
                f"{filename}: invalid risk_level"
            assert 0.0 <= result["shd_risk"] <= 1.0, \
                f"{filename}: shd_risk out of bounds: {result['shd_risk']}"


# ═══════════════════════════════════════════════════════════════════════════════
# 13. DIAGNOSIS ENGINE UNIT TESTS
# ═══════════════════════════════════════════════════════════════════════════════

class TestDiagnosisEngine:
    """Unit tests for the diagnosis engine in isolation."""

    @pytest.fixture(autouse=True)
    def setup(self):
        from app.diagnosis import DiagnosisEngine
        self.engine = DiagnosisEngine()

    def test_all_zeros_no_diagnoses(self):
        """Zero probabilities should produce no diagnoses."""
        probs = [0.0] * 12
        dxs = self.engine.diagnose(probs, {}, None)
        assert len(dxs) == 0, f"Zero probs produced diagnoses: {dxs}"

    def test_high_rv_dysfunction_triggers_rv_failure(self):
        probs = [0.0] * 12
        probs[7] = 0.85   # rv_dysfunction
        probs[11] = 0.70  # shd_composite
        dxs = self.engine.diagnose(probs, {}, None)
        names = [d["name"].lower() for d in dxs]
        assert any("rv" in n or "cor pulmonale" in n for n in names), \
            f"RV failure not diagnosed. Got: {names}"

    def test_high_pasp_triggers_pulmonary_hypertension(self):
        probs = [0.0] * 12
        probs[9]  = 0.85  # pasp_gte_45
        probs[10] = 0.80  # tr_max_gte_32
        probs[11] = 0.75  # shd_composite
        dxs = self.engine.diagnose(probs, {}, None)
        names = [d["name"].lower() for d in dxs]
        assert any("pulmonary" in n or "hypertension" in n for n in names), \
            f"PH not diagnosed. Got: {names}"

    def test_ptbxl_mi_triggers_mi_diagnosis(self):
        """High PTB-XL inferior MI probability should produce an MI diagnosis."""
        probs = [0.0] * 12
        ptbxl_probs = [0.0] * 12
        ptbxl_probs[1] = 0.85   # inferior_mi
        ptbxl_probs[11] = 0.90  # any_mi_composite
        dxs = self.engine.diagnose(probs, {}, ptbxl_probs)
        names = [d["name"].lower() for d in dxs]
        assert any("inferior" in n or "mi" in n for n in names), \
            f"Inferior MI not triggered. Got: {names}"

    def test_morphology_st_triggers_mi(self):
        """ST elevation in inferior leads should trigger inferior MI."""
        probs = [0.0] * 12
        morph = {
            "st_elevation_inferior": 0.55,
            "pathological_q_inferior": True,
            "ecg_mi_patterns": ["inferior"],
        }
        dxs = self.engine.diagnose(probs, morph, None)
        names = [d["name"].lower() for d in dxs]
        assert any("inferior" in n for n in names), \
            f"Inferior MI not detected from morphology. Got: {names}"

    def test_diagnoses_sorted_by_confidence(self):
        """Diagnoses must always be sorted highest confidence first."""
        probs = [0.5] * 12
        dxs = self.engine.diagnose(probs, {}, None)
        confs = [d["confidence"] for d in dxs]
        assert confs == sorted(confs, reverse=True), "Diagnoses not sorted by confidence"

    def test_source_field_present(self):
        """Every diagnosis must have a source field."""
        probs = [0.6] * 12
        dxs = self.engine.diagnose(probs, {}, None)
        for dx in dxs:
            assert "source" in dx, f"Missing source in: {dx}"
