"""
Clinician feedback integration for ECG Interpreter.

Set GOOGLE_FORM_URL to your published Google Form URL.
To find each field entry ID, open the form, right-click a field → Inspect,
and look for 'entry.XXXXXXXXX' in the name attribute.

Quick setup (5 min):
  1. Go to forms.google.com → New Form
  2. Add fields:
       - Case / File Name  (Short answer)  → ENTRY_CASE_ID
       - Overall Rating    (Linear scale 1-5) → ENTRY_RATING
       - Diagnosis Correct (Multiple choice: Yes / Partially / No) → ENTRY_DX_CORRECT
       - Comments          (Paragraph) → ENTRY_COMMENTS
  3. Click ⋮ → Get pre-filled link, fill dummy values, copy URL
  4. Replace the 'entry.XXXXXXXX' placeholders below with real IDs
  5. Replace GOOGLE_FORM_URL with your form's /formResponse URL
"""

import urllib.parse
import streamlit as st
import streamlit.components.v1 as components

# ── Configuration ──────────────────────────────────────────────────────────────
# Replace with your Google Form /viewform URL (for iframe preview)
GOOGLE_FORM_VIEWFORM_URL = "https://docs.google.com/forms/d/e/1FAIpQLSd3LCd0v_Du1Dcv3Ckb-bjx7PDMJ8jTHspbDMtd18X8sy8xLA/viewform?usp=pp_url"

# Replace with your Google Form /formResponse URL (for direct POST pre-fill)
GOOGLE_FORM_RESPONSE_URL = "https://docs.google.com/forms/d/e/1FAIpQLSd3LCd0v_Du1Dcv3Ckb-bjx7PDMJ8jTHspbDMtd18X8sy8xLA/viewform?usp=preview"

# Replace with actual entry IDs from your Google Form
ENTRY_NAME = "entry.2005620554"   # Name
ENTRY_MEDICAL_DOCTOR = "entry.1045781291"   # Medical Doctor
ENTRY_SPECIALTY = "entry.1065046570"   # Specialty
ENTRY_CASE_ID    = "entry.24172953"   # Case / File Name
ENTRY_DX_CORRECT = "entry.1166974658"   # Diagnosis Correct
ENTRY_RATING     = "entry.1020240997"   # Overall Rating (1–5)
ENTRY_COMMENTS   = "entry.839337160"   # Comments

def _build_prefill_url(name: str, medical_doctor: str, specialty: str, case_id: str, rating: int, dx_correct: str, comments: str) -> str:
    """Build a Google Form pre-filled URL that opens with user values already filled in."""
    params = {
        ENTRY_NAME: name,
        ENTRY_MEDICAL_DOCTOR: medical_doctor,
        ENTRY_SPECIALTY: specialty,
        ENTRY_CASE_ID:    case_id,
        ENTRY_RATING:     str(rating),
        ENTRY_DX_CORRECT: dx_correct,
        ENTRY_COMMENTS:   comments,
    }
    return GOOGLE_FORM_RESPONSE_URL + "?" + urllib.parse.urlencode(params)


def render_feedback_form(source_name: str = ""):
    """
    Render the clinician feedback section.
    Call this after _display_results() in app.py.
    """
    st.divider()
    with st.expander("📋 Submit Clinical Feedback", expanded=False):
        st.markdown(
            "**Help improve this tool** — rate the accuracy of the AI interpretation "
            "so we can refine the model over time."
        )
        with st.form("clinician_feedback", clear_on_submit=True):
            name = st.text_input("Your Name*", placeholder="e.g. Dr. Smith")
            medical_doctor = st.radio("Are you a medical doctor?*", options=["Yes", "No"], index=1, horizontal=True)
            specialty = st.text_input("Specialty*", placeholder="e.g. Cardiology")
            case_id = st.text_input(
                "Case / File name",
                value=source_name,
                placeholder="e.g. patient_001_ecg.pdf",
            )
            rating = st.slider(
                "Overall accuracy rating*",
                min_value=1, max_value=5, value=3,
                help="1 = Completely wrong  |  3 = Partially correct  |  5 = Fully correct",
            )
            dx_correct = st.radio(
                "Were the clinical diagnoses correct?*",
                options=["Yes", "Partially", "No"],
                horizontal=True,
            )
            comments = st.text_area(
                "Comments / corrections",
                placeholder="Describe any missed diagnoses, false positives, or suggestions...",
                height=100,
            )
            submitted = st.form_submit_button("Submit Feedback", type="primary")

        if submitted:
            # ── Required field validation ──────────────────────────────────
            errors = []
            if not name.strip():
                errors.append("**Your Name** is required.")
            if not specialty.strip():
                errors.append("**Specialty** is required.")

            if errors:
                for err in errors:
                    st.error(err)
            else:
                prefill_url = _build_prefill_url(
                    name=name.strip(),
                    medical_doctor=medical_doctor,
                    specialty=specialty.strip(),
                    case_id=case_id.strip(),
                    rating=rating,
                    dx_correct=dx_correct,
                    comments=comments.strip(),
                )
                st.success("Thank you for your feedback! Redirecting to the form…")
                # Open pre-filled Google Form in a new tab via JavaScript
                st.markdown(
                    f"""
                    <script>window.open("{prefill_url}", "_blank");</script>
                    <p>If the form did not open automatically,
                    <a href="{prefill_url}" target="_blank">click here</a>.</p>
                    """,
                    unsafe_allow_html=True,
                )
                # Also embed the pre-filled form inline
                _embed_form()


def _embed_form():
    """Embed the Google Form as an inline iframe."""
    embed_url = GOOGLE_FORM_VIEWFORM_URL + "?embedded=true"
    components.html(
        f'<iframe src="{embed_url}" width="100%" height="600" frameborder="0" '
        f'marginheight="0" marginwidth="0">Loading form…</iframe>',
        height=620,
        scrolling=True,
    )