"""
Clinical diagnosis engine for the ECG interpreter.

Maps the model's 12-condition probability outputs to likely clinical diagnoses
using evidence-weighted scoring. Each diagnosis has anchor findings (strong
drivers), supporting findings (additional confidence), and against findings
(dampeners).

Only diagnoses with enough evidence are returned, sorted by confidence.
"""

from __future__ import annotations
from typing import NamedTuple

import numpy as np


# ── PTB-XL label indices (matches PTBXLModel.LABEL_NAMES) ─────────────────────
PTBXL_IDX = {
    "normal_ecg":           0,
    "inferior_mi":          1,
    "anterior_mi":          2,
    "lateral_mi":           3,
    "posterior_mi":         4,
    "stt_changes":          5,
    "lbbb":                 6,
    "rbbb":                 7,
    "av_block":             8,
    "lvh":                  9,
    "atrial_fibrillation":  10,
    "any_mi_composite":     11,
}

# PTB-XL diagnosis rules: fired when model directly outputs a high probability
_PTBXL_RULES = {
    "Inferior Wall MI (ECG Model)": {
        "anchors":     ["inferior_mi", "any_mi_composite"],
        "description": (
            "The ECG diagnostic model trained on PTB-XL detects an inferior "
            "wall myocardial infarction pattern (leads II, III, aVF)."
        ),
    },
    "Anterior MI (ECG Model)": {
        "anchors":     ["anterior_mi", "any_mi_composite"],
        "description": (
            "The ECG diagnostic model detects an anterior wall MI pattern "
            "(leads V1–V4). LAD territory involvement likely."
        ),
    },
    "Lateral MI (ECG Model)": {
        "anchors":     ["lateral_mi", "any_mi_composite"],
        "description": (
            "The ECG diagnostic model detects a lateral wall MI pattern "
            "(leads I, aVL, V5–V6)."
        ),
    },
    "Posterior MI (ECG Model)": {
        "anchors":     ["posterior_mi", "any_mi_composite"],
        "description": (
            "The ECG diagnostic model detects posterior wall MI changes "
            "(dominant R in V1–V2, ST depression in anterior leads)."
        ),
    },
    "ST / T-wave Ischaemia (ECG Model)": {
        "anchors":     ["stt_changes"],
        "description": (
            "Diffuse ST segment or T-wave changes detected, consistent with "
            "myocardial ischaemia, electrolyte disturbance, or drug effect."
        ),
    },
    "Left Bundle Branch Block (ECG Model)": {
        "anchors":     ["lbbb"],
        "description": (
            "Complete or incomplete LBBB pattern detected. "
            "May mask underlying MI or ischaemia."
        ),
    },
    "Right Bundle Branch Block (ECG Model)": {
        "anchors":     ["rbbb"],
        "description": (
            "RBBB pattern detected — may be a normal variant or indicate "
            "right heart strain or structural disease."
        ),
    },
    "AV Conduction Block (ECG Model)": {
        "anchors":     ["av_block"],
        "description": (
            "AV conduction delay detected. Ranges from benign 1st-degree "
            "block to haemodynamically significant complete heart block."
        ),
    },
    "LV Hypertrophy — ECG Criteria (ECG Model)": {
        "anchors":     ["lvh"],
        "description": (
            "Voltage criteria for left ventricular hypertrophy met. "
            "Correlate with echo for confirmation."
        ),
    },
    "Atrial Fibrillation / Flutter (ECG Model)": {
        "anchors":     ["atrial_fibrillation"],
        "description": (
            "Irregular rhythm pattern consistent with atrial fibrillation "
            "or flutter. Rate control and anticoagulation assessment required."
        ),
    },
}

PTBXL_THRESHOLD = 0.40   # minimum probability to surface a PTB-XL diagnosis


def _score_ptbxl_rule(ptbxl_probs: np.ndarray, rule: dict) -> tuple[float, str, list]:
    """Return (confidence, level, findings) for a PTB-XL probability rule."""
    p_list = [float(ptbxl_probs[PTBXL_IDX[k]]) for k in rule["anchors"]
              if k in PTBXL_IDX]
    if not p_list:
        return 0.0, "POSSIBLE", []
    conf = max(p_list)
    if conf < PTBXL_THRESHOLD:
        return 0.0, "POSSIBLE", []

    from model.ptbxl_model import PTBXLModel
    findings = []
    for k in rule["anchors"]:
        idx = PTBXL_IDX.get(k)
        if idx is not None:
            p = float(ptbxl_probs[idx])
            if p >= PTBXL_THRESHOLD:
                findings.append(
                    f"{PTBXLModel.LABEL_DISPLAY[idx]} ({int(p*100)}%)"
                )

    level = "HIGH" if conf >= 0.65 else ("MODERATE" if conf >= 0.45 else "POSSIBLE")
    return round(conf, 3), level, findings


# ── EchoNext label indices (matches EchoNextModel.LABEL_NAMES) ────────────────
# Label key → index mapping (matches EchoNextModel.LABEL_NAMES)
IDX = {
    "lvef_lte_45":     0,
    "lvwt_gte_13":     1,
    "aortic_stenosis": 2,
    "aortic_regurg":   3,
    "mitral_regurg":   4,
    "tricuspid_regurg":5,
    "pulm_regurg":     6,
    "rv_dysfunction":  7,
    "pericardial_eff": 8,
    "pasp_gte_45":     9,
    "tr_max_gte_32":   10,
    "shd_composite":   11,
}


class _Rule(NamedTuple):
    """
    Definition of a clinical diagnosis rule.

    anchors:    list of (label_key, weight) — at least one must be elevated
    supporters: list of (label_key, weight) — boost confidence when present
    against:    list of (label_key, weight) — dampen when present and high
    threshold:  minimum anchor score to surface the diagnosis
    description: brief one-sentence clinical explanation
    """
    anchors:     list
    supporters:  list
    against:     list
    threshold:   float
    description: str


# ── Diagnosis rule book ────────────────────────────────────────────────────────

_RULES: dict[str, _Rule] = {

    "Systolic Heart Failure / DCM": _Rule(
        anchors=[
            ("lvef_lte_45", 1.0),
        ],
        supporters=[
            ("mitral_regurg",    0.4),   # functional MR from dilated annulus
            ("tricuspid_regurg", 0.3),   # functional TR
            ("rv_dysfunction",   0.4),   # biventricular failure
            ("shd_composite",    0.2),
        ],
        against=[
            ("aortic_stenosis", 0.5),    # AS explains reduced EF — separate dx
        ],
        threshold=0.35,
        description=(
            "Reduced left ventricular ejection force consistent with "
            "systolic dysfunction or dilated cardiomyopathy."
        ),
    ),

    "Aortic Stenosis": _Rule(
        anchors=[
            ("aortic_stenosis", 1.0),
        ],
        supporters=[
            ("lvwt_gte_13",  0.5),   # pressure-overload hypertrophy
            ("lvef_lte_45",  0.3),   # end-stage decompensation
            ("shd_composite",0.2),
        ],
        against=[
            ("aortic_regurg", 0.3),  # mixed valve disease — still show but dampen
        ],
        threshold=0.30,
        description=(
            "Obstruction of left ventricular outflow by a stenotic aortic valve, "
            "leading to pressure overload."
        ),
    ),

    "Aortic Regurgitation": _Rule(
        anchors=[
            ("aortic_regurg", 1.0),
        ],
        supporters=[
            ("lvef_lte_45",  0.4),   # volume-overload decompensation
            ("shd_composite",0.2),
        ],
        against=[
            ("aortic_stenosis", 0.3),
        ],
        threshold=0.30,
        description=(
            "Retrograde blood flow through the aortic valve during diastole, "
            "causing LV volume overload."
        ),
    ),

    "Mitral Valve Disease": _Rule(
        anchors=[
            ("mitral_regurg", 1.0),
        ],
        supporters=[
            ("lvef_lte_45",    0.3),   # functional MR or decompensation
            ("tricuspid_regurg",0.3),  # biatrial enlargement
            ("pasp_gte_45",    0.4),   # pulmonary hypertension from elevated LA pressure
            ("shd_composite",  0.2),
        ],
        against=[],
        threshold=0.30,
        description=(
            "Mitral valve regurgitation allowing backflow into the left atrium, "
            "causing volume overload and elevated LA pressure."
        ),
    ),

    "Pulmonary Hypertension": _Rule(
        anchors=[
            ("pasp_gte_45",    0.7),
            ("tr_max_gte_32",  0.7),
        ],
        supporters=[
            ("rv_dysfunction",  0.5),
            ("tricuspid_regurg",0.4),
            ("pulm_regurg",     0.3),
            ("shd_composite",   0.2),
        ],
        against=[
            ("lvef_lte_45",    0.2),   # LHF-driven PH — still show
        ],
        threshold=0.30,
        description=(
            "Elevated pulmonary arterial pressure causing right heart strain, "
            "detected via tricuspid regurgitant jet velocity and PASP estimate."
        ),
    ),

    "RV Failure / Cor Pulmonale": _Rule(
        anchors=[
            ("rv_dysfunction", 1.0),
        ],
        supporters=[
            ("tricuspid_regurg",0.5),
            ("pasp_gte_45",    0.5),
            ("tr_max_gte_32",  0.4),
            ("pulm_regurg",    0.3),
            ("shd_composite",  0.2),
        ],
        against=[
            ("lvef_lte_45",   0.3),   # biventricular failure — different entity
        ],
        threshold=0.30,
        description=(
            "Right ventricular systolic dysfunction, often from chronic "
            "pulmonary hypertension or cor pulmonale."
        ),
    ),

    "Pericardial Disease": _Rule(
        anchors=[
            ("pericardial_eff", 1.0),
        ],
        supporters=[
            ("rv_dysfunction",  0.3),  # tamponade physiology
            ("shd_composite",   0.2),
        ],
        against=[],
        threshold=0.30,
        description=(
            "Moderate-to-large pericardial effusion; may indicate pericarditis, "
            "malignancy, hypothyroidism, or haemopericardium."
        ),
    ),

    "LV Hypertrophy / Hypertensive Heart Disease": _Rule(
        anchors=[
            ("lvwt_gte_13", 1.0),
        ],
        supporters=[
            ("aortic_stenosis", 0.3),  # pressure-overload overlap
            ("shd_composite",   0.2),
        ],
        against=[
            ("lvef_lte_45", 0.4),     # dilated → wall thinning, not hypertrophy
        ],
        threshold=0.30,
        description=(
            "Increased left ventricular wall thickness consistent with pressure "
            "overload (hypertension, aortic stenosis) or hypertrophic cardiomyopathy."
        ),
    ),

    "Mixed Valvular Disease": _Rule(
        anchors=[
            ("aortic_stenosis",  0.5),
            ("aortic_regurg",    0.5),
            ("mitral_regurg",    0.5),
            ("tricuspid_regurg", 0.5),
        ],
        supporters=[
            ("shd_composite", 0.3),
        ],
        against=[],
        threshold=0.60,   # high bar: need TWO distinct valvular findings elevated
        description=(
            "Multiple valvular lesions present simultaneously, likely from "
            "rheumatic heart disease or degenerative valve changes."
        ),
    ),
}


# ── ECG morphology MI rules (independent of the 12 model probabilities) ───────

# Confidence scoring for morphology-based diagnoses:
#   ST elevation + Q waves → HIGH (0.80–0.90)
#   ST elevation alone     → MODERATE-HIGH (0.60–0.75)
#   Q waves alone          → MODERATE (0.55 — old/resolved MI)

_MORPH_RHYTHM_RULES = {
    "Irregular Rhythm / Atrial Fibrillation": {
        "rr_cv_key":   "rr_cv",
        "flag_key":    "irregular_rhythm",
        "description": (
            "Highly irregular R-R intervals detected (coefficient of variation > 0.15), "
            "consistent with atrial fibrillation, atrial flutter with variable block, "
            "or frequent ectopic beats. Rate control and anticoagulation should be assessed."
        ),
    },
    "Wide QRS Complex / Bundle Branch Block": {
        "qrs_ms_key":  "qrs_duration_ms",
        "flag_key":    "wide_qrs",
        "description": (
            "QRS duration > 115 ms detected, consistent with left or right bundle branch "
            "block, ventricular conduction delay, or pacemaker-mediated rhythm."
        ),
    },
}


def _score_morph_rhythm_rule(ecg_morph: dict, rule: dict) -> tuple[float, str, list[str]]:
    """Score rhythm/conduction rules based on morphological flags."""
    flag = ecg_morph.get(rule["flag_key"], False)
    findings = []
    if not flag:
        return 0.0, "POSSIBLE", []

    if "rr_cv_key" in rule:
        rr_cv = ecg_morph.get(rule["rr_cv_key"])
        conf = min(0.85, 0.50 + float(rr_cv) * 1.5) if rr_cv else 0.65
        findings.append(f"R-R CV = {rr_cv:.3f}" if rr_cv else "Irregular R-R intervals")
    elif "qrs_ms_key" in rule:
        qrs_ms = ecg_morph.get(rule["qrs_ms_key"])
        conf = min(0.82, 0.50 + max(0, float(qrs_ms) - 115) / 100) if qrs_ms else 0.65
        findings.append(f"QRS = {qrs_ms:.0f} ms" if qrs_ms else "Wide QRS complex")
    else:
        conf = 0.65

    level = "HIGH" if conf >= 0.65 else "MODERATE"
    return round(conf, 3), level, findings


_MORPH_RULES = {
    "Inferior Wall MI / Ischaemia": {
        "territory":    "inferior",
        "st_key":       "st_elevation_inferior",
        "q_key":        "pathological_q_inferior",
        "description": (
            "ST elevation and/or pathological Q waves in the inferior leads "
            "(II, III, aVF), consistent with inferior wall myocardial infarction "
            "or acute ischaemia. Right ventricular involvement should be excluded."
        ),
    },
    "Anterior MI / Ischaemia": {
        "territory":    "anterior",
        "st_key":       "st_elevation_anterior",
        "q_key":        "pathological_q_anterior",
        "description": (
            "ST elevation and/or pathological Q waves in the anterior leads "
            "(V1–V4), consistent with anterior myocardial infarction. "
            "LAD territory involvement likely."
        ),
    },
    "Lateral Ischaemia": {
        "territory":    "lateral",
        "st_key":       "st_elevation_lateral",
        "q_key":        None,
        "description": (
            "ST changes in lateral leads (I, aVL, V5, V6), suggesting lateral "
            "wall ischaemia or as reciprocal changes to an inferior or posterior MI."
        ),
    },
}

ST_THRESHOLD = 0.20   # normalised units — empirical ≈1 mm on a typical ECG


def _score_morph_rule(ecg_morph: dict, rule: dict) -> tuple[float, str, list[str]]:
    """Return (confidence, level, findings_list) for a morphology-based rule."""
    st_val  = ecg_morph.get(rule["st_key"]) or 0.0
    has_q   = ecg_morph.get(rule["q_key"], False) if rule["q_key"] else False
    findings = []

    if st_val > ST_THRESHOLD:
        findings.append(
            f"ST elevation in {rule['territory']} leads ({st_val:.2f} units)"
        )
    if has_q:
        findings.append(f"Pathological Q waves in {rule['territory']} leads")

    if st_val > ST_THRESHOLD and has_q:
        conf = min(0.90, 0.65 + st_val * 0.40)
    elif st_val > ST_THRESHOLD:
        conf = min(0.78, 0.50 + st_val * 0.40)
    elif has_q:
        conf = 0.58
    else:
        conf = 0.0

    if conf >= 0.65:
        level = "HIGH"
    elif conf >= 0.45:
        level = "MODERATE"
    else:
        level = "POSSIBLE"

    return round(conf, 3), level, findings


# ── Scoring logic ──────────────────────────────────────────────────────────────

def _anchor_score(probs: np.ndarray, anchors: list) -> float:
    """
    Max-pooled anchor score: the strongest single anchor drives this.
    Weighted by the anchor's weight × its probability.
    """
    scores = [probs[IDX[key]] * w for key, w in anchors]
    return float(max(scores)) if scores else 0.0


def _support_bonus(probs: np.ndarray, supporters: list) -> float:
    """
    Sum of support contributions, capped at 0.25 to avoid overshooting.
    """
    bonus = sum(probs[IDX[key]] * w for key, w in supporters)
    return min(float(bonus), 0.25)


def _against_dampen(probs: np.ndarray, against: list) -> float:
    """
    Multiplicative dampener in [0.5, 1.0].
    High against-findings reduce confidence but never to zero.
    """
    dampen = 1.0
    for key, w in against:
        dampen *= max(0.5, 1.0 - probs[IDX[key]] * w)
    return float(dampen)


def _confidence(probs: np.ndarray, rule: _Rule) -> float:
    """
    Composite confidence score ∈ [0, 1].
    anchor_score + support_bonus, dampened by against findings.
    """
    anchor  = _anchor_score(probs, rule.anchors)
    support = _support_bonus(probs, rule.supporters)
    dampen  = _against_dampen(probs, rule.against)
    raw     = min(anchor + support, 1.0) * dampen
    return round(float(raw), 3)


def _supporting_findings(probs: np.ndarray, rule: _Rule, label_names: list[str]) -> list[str]:
    """Return human-readable list of findings that are elevated and relevant."""
    from model.architecture import EchoNextModel
    display = EchoNextModel.LABEL_DISPLAY
    keys    = EchoNextModel.LABEL_NAMES

    relevant_keys = {k for k, _ in rule.anchors + rule.supporters}
    findings = []
    for key, _ in rule.anchors + rule.supporters:
        idx = IDX[key]
        if probs[idx] >= 0.30:
            findings.append(f"{display[idx]} ({int(probs[idx]*100)}%)")
    # Deduplicate while preserving order
    seen = set()
    deduped = []
    for f in findings:
        if f not in seen:
            seen.add(f)
            deduped.append(f)
    return deduped


# ── Public API ─────────────────────────────────────────────────────────────────

class DiagnosisEngine:
    """
    Maps model output probabilities to clinical diagnoses.

    Usage:
        engine = DiagnosisEngine()
        diagnoses = engine.diagnose(probs_list)
    """

    def diagnose(self, probabilities: list[float],
                 ecg_morph: dict = None,
                 ptbxl_probs: list[float] = None) -> list[dict]:
        """
        Args:
            probabilities: list of 12 floats from model output (0–1)
            ecg_morph:     dict of morphological ECG features (ST levels, Q waves)

        Returns:
            List of diagnosis dicts sorted by confidence (highest first).
            Morphology-based diagnoses (MI patterns) are flagged with
            source='ecg_morphology' and appear before probability-based ones
            at the same confidence level.

            Each dict:
                name         -> str
                confidence   -> float (0–1)
                pct          -> int (percentage)
                description  -> str
                findings     -> list[str]
                level        -> str  'HIGH' / 'MODERATE' / 'POSSIBLE'
                source       -> str  'model_probabilities' | 'ecg_morphology'
        """
        probs      = np.array(probabilities, dtype=np.float32)
        morph      = ecg_morph or {}
        ptbxl_arr  = np.array(ptbxl_probs, dtype=np.float32) if ptbxl_probs else None
        results    = []

        # ── PTB-XL model diagnoses (direct ECG pattern recognition) ─────────────
        if ptbxl_arr is not None:
            for name, rule in _PTBXL_RULES.items():
                conf, level, findings = _score_ptbxl_rule(ptbxl_arr, rule)
                if conf < PTBXL_THRESHOLD:
                    continue
                results.append({
                    "name":        name,
                    "confidence":  conf,
                    "pct":         int(conf * 100),
                    "description": rule["description"],
                    "findings":    findings,
                    "level":       level,
                    "source":      "ptbxl_model",
                })

        # ── Morphology-based rhythm / conduction diagnoses ──────────────────────
        for name, rule in _MORPH_RHYTHM_RULES.items():
            conf, level, findings = _score_morph_rhythm_rule(morph, rule)
            if conf < 0.40:
                continue
            results.append({
                "name":        name,
                "confidence":  conf,
                "pct":         int(conf * 100),
                "description": rule["description"],
                "findings":    findings,
                "level":       level,
                "source":      "ecg_morphology",
            })

        # ── Morphology-based MI diagnoses (highest diagnostic specificity) ──────
        for name, rule in _MORPH_RULES.items():
            conf, level, findings = _score_morph_rule(morph, rule)
            if conf < 0.40:
                continue
            results.append({
                "name":        name,
                "confidence":  conf,
                "pct":         int(conf * 100),
                "description": rule["description"],
                "findings":    findings,
                "level":       level,
                "source":      "ecg_morphology",
            })

        # ── Model-probability-based structural diagnoses ──────────────────────
        for name, rule in _RULES.items():
            conf = _confidence(probs, rule)
            anchor_max = max(probs[IDX[key]] for key, _ in rule.anchors)
            if conf < rule.threshold and anchor_max < 0.70:
                continue

            if name == "Mixed Valvular Disease":
                n_elevated = sum(
                    1 for key, _ in rule.anchors if probs[IDX[key]] >= 0.30
                )
                if n_elevated < 2:
                    continue

            findings = _supporting_findings(probs, rule, [])

            if conf >= 0.60:
                level = "HIGH"
            elif conf >= 0.40:
                level = "MODERATE"
            else:
                level = "POSSIBLE"

            results.append({
                "name":        name,
                "confidence":  conf,
                "pct":         int(conf * 100),
                "description": rule.description,
                "findings":    findings,
                "level":       level,
                "source":      "model_probabilities",
            })

        # Sort: by confidence desc; within same tier PTB-XL first, morph second, model last
        _source_order = {"ptbxl_model": 0, "ecg_morphology": 1, "model_probabilities": 2}
        results.sort(key=lambda x: (
            -x["confidence"],
            _source_order.get(x["source"], 3),
        ))
        return results
