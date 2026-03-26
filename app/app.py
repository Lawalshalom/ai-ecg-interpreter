"""
ECG Interpreter — Streamlit App

Run with:
    streamlit run app/app.py
"""

import os
import sys
import io
import time

import numpy as np
import streamlit as st
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from digitizer.pipeline import ECGDigitizer
from app.inference import ECGInferenceEngine
from app.feedback import render_feedback_form

# ── Page config ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="ECG Interpreter",
    page_icon="🫀",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Theme definitions ───────────────────────────────────────────────────────────
_LIGHT_CSS = """
<style>
    .stApp { background-color: #ffffff; color: #1a1a2e; }
    .risk-high     { background:#ffe0e0; border-left:5px solid #d32f2f; padding:12px 16px; border-radius:4px; }
    .risk-moderate { background:#fff8e1; border-left:5px solid #f57c00; padding:12px 16px; border-radius:4px; }
    .risk-low      { background:#e8f5e9; border-left:5px solid #388e3c; padding:12px 16px; border-radius:4px; }
    .disclaimer    { background:#f5f5f5; border:1px solid #ccc; padding:10px 14px; border-radius:4px; font-size:0.8em; color:#555; }
    .feature-card  { background:#fafafa; border:1px solid #e0e0e0; border-radius:6px; padding:10px; text-align:center; }
    .section-title { font-size:1.1em; font-weight:600; color:#1a237e; margin-bottom:8px; }
    .dx-card-high     { background:#fff0f0; border:1.5px solid #d32f2f; border-radius:8px; padding:12px 16px; margin-bottom:10px; }
    .dx-card-moderate { background:#fff8e1; border:1.5px solid #f57c00; border-radius:8px; padding:12px 16px; margin-bottom:10px; }
    .dx-card-possible { background:#f3f6fd; border:1.5px solid #1565c0; border-radius:8px; padding:12px 16px; margin-bottom:10px; }
    .dx-name   { font-size:1.05em; font-weight:700; }
    .dx-pct    { font-size:1.4em; font-weight:800; }
    .dx-desc   { font-size:0.82em; color:#444; margin-top:4px; }
    .dx-findings { font-size:0.80em; color:#555; margin-top:6px; }
</style>
"""

_DARK_CSS = """
<style>
    .stApp, [data-testid="stAppViewContainer"] {
        background-color: #0f1117 !important; color: #e8eaf6 !important;
    }
    [data-testid="stSidebar"] { background-color: #1a1d27 !important; }
    [data-testid="stHeader"]  { background-color: #0f1117 !important; }
    .stMarkdown, .stText, label, p, span { color: #e8eaf6 !important; }
    .stTextInput > div > div > input,
    .stTextArea  > div > div > textarea,
    .stSelectbox > div > div > div { background:#1e2130 !important; color:#e8eaf6 !important; border-color:#3949ab !important; }
    .risk-high     { background:#3b1a1a; border-left:5px solid #ef5350; padding:12px 16px; border-radius:4px; }
    .risk-moderate { background:#3b2f1a; border-left:5px solid #ffa726; padding:12px 16px; border-radius:4px; }
    .risk-low      { background:#1a3b22; border-left:5px solid #66bb6a; padding:12px 16px; border-radius:4px; }
    .disclaimer    { background:#1e2130; border:1px solid #3949ab; padding:10px 14px; border-radius:4px; font-size:0.8em; color:#b0bec5; }
    .feature-card  { background:#1e2130; border:1px solid #3949ab; border-radius:6px; padding:10px; text-align:center; }
    .section-title { font-size:1.1em; font-weight:600; color:#7986cb; margin-bottom:8px; }
    .dx-card-high     { background:#2d1515; border:1.5px solid #ef5350; border-radius:8px; padding:12px 16px; margin-bottom:10px; }
    .dx-card-moderate { background:#2d2315; border:1.5px solid #ffa726; border-radius:8px; padding:12px 16px; margin-bottom:10px; }
    .dx-card-possible { background:#151d2d; border:1.5px solid #5c6bc0; border-radius:8px; padding:12px 16px; margin-bottom:10px; }
    .dx-name   { font-size:1.05em; font-weight:700; }
    .dx-pct    { font-size:1.4em; font-weight:800; }
    .dx-desc   { font-size:0.82em; color:#b0bec5; margin-top:4px; }
    .dx-findings { font-size:0.80em; color:#90a4ae; margin-top:6px; }
    hr { border-color: #3949ab !important; }
</style>
"""


def _apply_theme(dark: bool):
    st.markdown(_DARK_CSS if dark else _LIGHT_CSS, unsafe_allow_html=True)


# ── Cached resources ───────────────────────────────────────────────────────────
@st.cache_resource(show_spinner="Loading ECG interpreter model...")
def load_engine():
    return ECGInferenceEngine()


@st.cache_resource(show_spinner=False)
def load_digitizer():
    return ECGDigitizer()


# ── Helper functions ───────────────────────────────────────────────────────────

def plot_12_lead_ecg(waveform: np.ndarray, title: str = "Digitized ECG") -> go.Figure:
    """
    Plot 12-lead ECG as a clinical-style grid layout.
    waveform: (2500, 12) float32
    """
    LEAD_NAMES = ["I", "II", "III", "aVR", "aVL", "aVF", "V1", "V2", "V3", "V4", "V5", "V6"]
    # Layout: 3 rows × 4 cols
    layout_order = [
        [(0, "I"), (1, "aVR"), (2, "V1"), (3, "V4")],
        [(4, "II"), (5, "aVL"), (6, "V2"), (7, "V5")],
        [(8, "III"), (9, "aVF"), (10, "V3"), (11, "V6")],
    ]
    lead_idx_map = {name: i for i, name in enumerate(LEAD_NAMES)}

    fig = make_subplots(
        rows=3, cols=4,
        shared_xaxes=True,
        vertical_spacing=0.06,
        horizontal_spacing=0.04,
        subplot_titles=[
            "I", "aVR", "V1", "V4",
            "II", "aVL", "V2", "V5",
            "III", "aVF", "V3", "V6",
        ],
    )

    DISPLAY_SAMPLES = 625  # 2.5 s → 3–4 beats at typical resting HR
    t = np.linspace(0, 2.5, DISPLAY_SAMPLES)
    row_col = [
        (1, 1), (1, 2), (1, 3), (1, 4),
        (2, 1), (2, 2), (2, 3), (2, 4),
        (3, 1), (3, 2), (3, 3), (3, 4),
    ]
    plot_order = [0, 3, 6, 9, 1, 4, 7, 10, 2, 5, 8, 11]

    for plot_pos, lead_idx in enumerate(plot_order):
        row, col = row_col[plot_pos]
        fig.add_trace(
            go.Scatter(
                x=t,
                y=waveform[:DISPLAY_SAMPLES, lead_idx],
                mode="lines",
                line=dict(color="#c62828", width=1.0),
                name=LEAD_NAMES[lead_idx],
                showlegend=False,
            ),
            row=row, col=col,
        )

    fig.update_layout(
        title=dict(text=title, font=dict(size=14, color="#1a237e")),
        height=420,
        margin=dict(l=40, r=20, t=60, b=20),
        paper_bgcolor="white",
        plot_bgcolor="#fff9f9",
    )
    fig.update_xaxes(showgrid=True, gridcolor="#ffcdd2", gridwidth=0.5, ticksuffix="s")
    fig.update_yaxes(showgrid=True, gridcolor="#ffcdd2", gridwidth=0.5, showticklabels=False)
    return fig


def risk_badge_html(risk_level: str, shd_prob: float) -> str:
    color_map = {"HIGH": "#d32f2f", "MODERATE": "#f57c00", "LOW": "#388e3c"}
    bg_map    = {"HIGH": "#ffe0e0", "MODERATE": "#fff8e1", "LOW": "#e8f5e9"}
    color = color_map.get(risk_level, "#555")
    bg    = bg_map.get(risk_level, "#eee")
    pct   = int(shd_prob * 100)
    return f"""
    <div style="background:{bg}; border:2px solid {color}; border-radius:8px;
                padding:16px 20px; text-align:center; margin-bottom:12px;">
        <div style="font-size:2em; font-weight:700; color:{color};">{risk_level}</div>
        <div style="font-size:1.1em; color:{color};">Structural Heart Disease Risk</div>
        <div style="font-size:2.5em; font-weight:800; color:{color};">{pct}%</div>
        <div style="font-size:0.9em; color:#555;">Composite SHD Probability</div>
    </div>
    """


def diagnosis_card_html(dx: dict) -> str:
    level  = dx["level"].lower()
    color_map = {"high": "#d32f2f", "moderate": "#f57c00", "possible": "#1565c0"}
    color  = color_map.get(level, "#555")

    source = dx.get("source", "model_probabilities")
    if source == "ptbxl_model":
        source_badge = (
            '<span style="background:#4a148c; color:#fff; font-size:0.68em; '
            'padding:2px 6px; border-radius:3px; margin-left:6px; '
            'vertical-align:middle;">PTB-XL MODEL</span>'
        )
    elif source == "ecg_morphology":
        source_badge = (
            '<span style="background:#1a237e; color:#fff; font-size:0.68em; '
            'padding:2px 6px; border-radius:3px; margin-left:6px; '
            'vertical-align:middle;">ECG PATTERN</span>'
        )
    else:
        source_badge = ""

    findings_html = ""
    if dx["findings"]:
        items = "  &nbsp;·&nbsp;  ".join(dx["findings"])
        findings_html = f'<div class="dx-findings"><b>Supporting findings:</b> {items}</div>'

    return f"""
    <div class="dx-card-{level}">
        <div style="display:flex; justify-content:space-between; align-items:center;">
            <span class="dx-name" style="color:{color};">{dx['name']}{source_badge}</span>
            <span class="dx-pct" style="color:{color};">{dx['pct']}%</span>
        </div>
        <div style="font-size:0.78em; color:{color}; font-weight:500;">{dx['level']} CONFIDENCE</div>
        <div class="dx-desc">{dx['description']}</div>
        {findings_html}
    </div>
    """


def prob_bar_html(label: str, prob: float) -> str:
    pct = int(prob * 100)
    if pct >= 60:
        color = "#d32f2f"
    elif pct >= 35:
        color = "#f57c00"
    else:
        color = "#388e3c"
    bar_width = max(2, pct)
    return f"""
    <div style="margin-bottom:6px;">
        <div style="display:flex; justify-content:space-between; margin-bottom:2px;">
            <span style="font-size:0.85em;">{label}</span>
            <span style="font-size:0.85em; font-weight:600; color:{color};">{pct}%</span>
        </div>
        <div style="background:#eee; border-radius:4px; height:8px;">
            <div style="width:{bar_width}%; background:{color}; border-radius:4px; height:8px;"></div>
        </div>
    </div>
    """


# ── Sidebar ────────────────────────────────────────────────────────────────────

def render_sidebar(engine: ECGInferenceEngine) -> tuple[dict, bool]:
    with st.sidebar:
        # ── Theme toggle ───────────────────────────────────────────────────────
        dark_mode = st.toggle("Dark theme", value=False, key="dark_mode")

        st.divider()
        st.markdown("### Patient Demographics")
        st.markdown("*Optional — improves accuracy*")

        age = st.number_input("Age (years)", min_value=18, max_value=100, value=40, step=1)
        sex = st.selectbox("Sex", ["Male", "Female"])
        sex_val = 1 if sex == "Male" else 0

        st.divider()
        st.markdown("### Model Status")

        # EchoNext structural model
        if engine.checkpoint_loaded:
            auroc = getattr(engine, "val_auroc", None)
            st.success("EchoNext model loaded")
            if auroc:
                st.metric("EchoNext Val AUROC", f"{auroc:.4f}")
        else:
            st.warning("EchoNext model not trained.\n\n"
                       "```\npython training/train.py\n```")

        # PTB-XL ECG diagnostic model
        from app.inference import _ptbxl_engine
        if _ptbxl_engine.checkpoint_loaded:
            st.success("PTB-XL model loaded")
            if _ptbxl_engine.val_auroc:
                st.metric("PTB-XL Val AUROC", f"{_ptbxl_engine.val_auroc:.4f}")
        else:
            st.info("PTB-XL model not yet trained.\n\n"
                    "```\npython training/ptbxl_download.py\n"
                    "python training/ptbxl_train.py\n```")

        st.divider()
        st.markdown("""
        <div class="disclaimer">
        <b>Medical Disclaimer</b><br>
        This tool is for <b>research and educational use only</b>.
        It does not constitute medical advice and must not replace
        clinical evaluation by a qualified cardiologist.
        All outputs should be verified by a licensed healthcare professional.
        </div>
        """, unsafe_allow_html=True)

    return {"age": age, "sex": sex_val}, dark_mode


# ── Main app ───────────────────────────────────────────────────────────────────

def main():
    st.title("ECG Structural Heart Disease Interpreter")
    st.caption("AI-powered 12-lead ECG analysis using the EchoNext deep learning model")

    engine    = load_engine()
    digitizer = load_digitizer()
    patient_info_base, dark_mode = render_sidebar(engine)
    _apply_theme(dark_mode)

    # ── Upload section ─────────────────────────────────────────────────────────
    tab_upload, tab_demo = st.tabs(["Upload ECG", "Demo / Sample ECG"])

    with tab_upload:
        st.markdown("#### Upload ECG Image")
        st.markdown(
            "Supported formats: **JPEG, PNG, PDF** — scanned paper ECGs, digital ECGs, or smartphone photos of ECG printouts."
        )
        uploaded = st.file_uploader(
            "Choose ECG file",
            type=["jpg", "jpeg", "png", "pdf", "bmp", "tiff"],
            label_visibility="collapsed",
        )

        if uploaded is not None:
            file_bytes = uploaded.read()

            col_img, col_pad = st.columns([1, 1])
            with col_img:
                if uploaded.type != "application/pdf":
                    pil_img = Image.open(io.BytesIO(file_bytes))
                    st.image(pil_img, caption="Uploaded ECG", use_container_width=True)

            if st.button("Analyze ECG", type="primary", use_container_width=True):
                _run_analysis(file_bytes, patient_info_base, digitizer, engine, source_name=uploaded.name)

    with tab_demo:
        st.markdown("#### Sample ECG from PTB-XL Database")
        st.info(
            "This loads a real ECG record from the public PTB-XL database via WFDB. "
            "Use this to verify the pipeline end-to-end."
        )
        if st.button("Load & Analyze Sample ECG", type="primary"):
            from digitizer.pipeline import load_demo_ecg
            with st.spinner("Loading sample ECG..."):
                waveform, meta = load_demo_ecg()

            _display_results(
                waveform=waveform,
                meta=meta,
                result=engine.predict(waveform, patient_info_base),
                source_name=meta.get("source", "Demo ECG"),
            )


def _run_analysis(file_bytes: bytes, patient_info: dict, digitizer: ECGDigitizer,
                  engine: ECGInferenceEngine, source_name: str):
    """Digitize uploaded file and run inference."""
    col1, col2 = st.columns(2)

    with col1:
        with st.spinner("Digitizing ECG image..."):
            t0 = time.time()
            waveform, meta = digitizer.process(io.BytesIO(file_bytes))
            digitize_time = time.time() - t0

    if not meta.get("success", False):
        err = meta.get("error", "Unknown error")
        # Distinguish user-facing validation rejections from processing failures
        validation_phrases = [
            "does not contain a recognizable",
            "portrait-oriented document",
            "landscape-oriented",
            "appears blank",
            "too dark",
            "too small",
        ]
        is_validation_error = any(p in err for p in validation_phrases)
        if is_validation_error:
            st.warning(f"⚠️ {err}")
        else:
            st.error(f"ECG digitization failed: {err}")
        if meta.get("notes"):
            with st.expander("Processing notes"):
                for note in meta["notes"]:
                    st.write(note)
        return

    with col2:
        with st.spinner("Running AI analysis..."):
            t0 = time.time()
            result = engine.predict(waveform, patient_info)
            infer_time = time.time() - t0

    st.caption(f"Digitized in {digitize_time:.1f}s | Inference in {infer_time:.2f}s")
    _display_results(waveform, meta, result, source_name)


def _display_results(waveform: np.ndarray, meta: dict, result: dict, source_name: str):
    """Render the full analysis result panel."""
    st.divider()
    st.markdown(f"### Analysis Results — *{source_name}*")

    # ── Row 1: Risk badge + ECG features ──────────────────────────────────────
    col_risk, col_features = st.columns([1, 2])

    with col_risk:
        st.markdown(
            risk_badge_html(result["risk_level"], result["shd_risk"]),
            unsafe_allow_html=True,
        )

    with col_features:
        st.markdown('<div class="section-title">Extracted ECG Features</div>', unsafe_allow_html=True)
        raw = result.get("raw_features", {})
        feats = result.get("ecg_features", {})
        f_col1, f_col2, f_col3 = st.columns(3)
        def metric_val(key, unit="", decimals=0):
            v = raw.get(key) or feats.get(key)
            if v is None:
                return "N/A"
            return f"{v:.{decimals}f}{unit}"

        f_col1.metric("Heart Rate", metric_val("ventricular_rate", " bpm"))
        f_col1.metric("Age", metric_val("age_at_ecg", " yrs"))
        f_col2.metric("QRS Duration", metric_val("qrs_duration", " ms"))
        f_col2.metric("QTc Interval", metric_val("qt_corrected", " ms"))
        f_col3.metric("PR Interval", metric_val("pr_interval", " ms"))
        f_col3.metric("Sex", "Male" if raw.get("sex", 1) == 1 else "Female")

    # ── Row 2: ECG waveform plot ───────────────────────────────────────────────
    st.markdown('<div class="section-title">Digitized Waveform (12 Leads)</div>', unsafe_allow_html=True)
    leads_ok = meta.get("n_leads_detected", 0)
    if leads_ok < 12:
        st.warning(f"Only {leads_ok}/12 leads detected — results may be less accurate.")

    w = waveform.squeeze()
    if w.ndim == 2 and w.shape[1] == 12:
        fig = plot_12_lead_ecg(w, title="Digitized 12-Lead ECG")
    elif w.ndim == 2 and w.shape[0] == 12:
        fig = plot_12_lead_ecg(w.T, title="Digitized 12-Lead ECG")
    else:
        fig = None

    if fig:
        st.plotly_chart(fig, use_container_width=True)

    # ── Row 3: Per-condition probabilities ────────────────────────────────────
    st.divider()
    st.markdown('<div class="section-title">Per-Condition Analysis</div>', unsafe_allow_html=True)

    probs  = result["probabilities"]
    labels = result["labels"]

    # Composite SHD at the top
    st.markdown(prob_bar_html(f"**{labels[11]}**", probs[11]), unsafe_allow_html=True)
    st.markdown("<hr style='margin:6px 0;border-color:#eee;'>", unsafe_allow_html=True)

    col_left, col_right = st.columns(2)
    half = 11 // 2
    with col_left:
        for i in range(half):
            st.markdown(prob_bar_html(labels[i], probs[i]), unsafe_allow_html=True)
    with col_right:
        for i in range(half, 11):
            st.markdown(prob_bar_html(labels[i], probs[i]), unsafe_allow_html=True)

    # ── Row 4: Top findings ────────────────────────────────────────────────────
    if result["top_findings"]:
        st.divider()
        st.markdown('<div class="section-title">Notable Findings (> 30% probability)</div>', unsafe_allow_html=True)
        for finding in result["top_findings"]:
            pct = int(finding["probability"] * 100)
            icon = "🔴" if pct >= 60 else "🟠"
            st.markdown(f"{icon} **{finding['condition']}** — {pct}% probability")

    # ── Row 5: Clinical Interpretation ────────────────────────────────────────
    diagnoses       = result.get("diagnoses", [])
    normal_ecg_prob = result.get("normal_ecg_prob", 0.0)
    st.divider()
    st.markdown('<div class="section-title">Clinical Interpretation</div>', unsafe_allow_html=True)

    # Normal ECG card — only shown when ALL individual findings are ≤ 50%
    # AND no ECG morphological abnormality (ST elevation / Q waves) detected
    probs_list    = result["probabilities"]
    max_individual = max(probs_list[:11])
    mi_patterns   = result.get("ecg_morph", {}).get("ecg_mi_patterns", [])
    if max_individual <= 0.50 and not mi_patterns:
        n_pct = int(normal_ecg_prob * 100)
        st.markdown(f"""
        <div style="background:#e8f5e9; border:2px solid #388e3c; border-radius:8px;
                    padding:12px 16px; margin-bottom:14px; display:flex;
                    justify-content:space-between; align-items:center;">
            <div>
                <div style="font-size:1.05em; font-weight:700; color:#388e3c;">Normal ECG</div>
                <div style="font-size:0.78em; color:#388e3c; font-weight:500;">NO SIGNIFICANT STRUCTURAL ABNORMALITY DETECTED</div>
                <div style="font-size:0.82em; color:#444; margin-top:4px;">
                    All individual condition probabilities are ≤ 50%. This tracing is
                    likely within normal limits for structural heart disease.
                </div>
            </div>
            <div style="font-size:2.2em; font-weight:800; color:#388e3c; padding-left:20px;">{n_pct}%</div>
        </div>
        """, unsafe_allow_html=True)

    if diagnoses:
        st.caption(
            "Potential diagnoses are inferred from the pattern of elevated findings. "
            "Confidence reflects how well the probability profile fits each condition. "
            "Multiple diagnoses may coexist."
        )
        dx_col1, dx_col2 = st.columns(2)
        for i, dx in enumerate(diagnoses):
            col = dx_col1 if i % 2 == 0 else dx_col2
            with col:
                st.markdown(diagnosis_card_html(dx), unsafe_allow_html=True)
    else:
        st.info(
            "No specific structural diagnosis pattern identified from the model output. "
            "All individual condition probabilities are below the clinical threshold."
        )

    # ── Row 6: Processing notes ────────────────────────────────────────────────
    if meta.get("notes"):
        with st.expander("Digitization processing notes"):
            for note in meta["notes"]:
                st.caption(note)

    # ── Row 7: Clinician feedback ──────────────────────────────────────────────
    render_feedback_form(source_name=source_name)

    # ── Disclaimer ────────────────────────────────────────────────────────────
    st.divider()
    st.markdown("""
    <div class="disclaimer">
    <b>Important Medical Disclaimer:</b> This AI tool is intended for <b>research and educational purposes only</b>.
    Results must not be used for clinical decision-making without review by a qualified cardiologist.
    The model is trained on EchoNext data from Columbia University Irving Medical Center and may not
    generalize to all patient populations or ECG recording systems.
    Sensitivity and specificity vary by condition — always confirm findings with echocardiography.
    </div>
    """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()
