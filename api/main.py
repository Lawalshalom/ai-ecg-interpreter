"""
ECG Interpreter REST API
Wraps the existing Python inference engine for mobile clients.
"""
import sys
import os
import traceback
import io
from typing import Optional

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from fastapi import FastAPI, File, UploadFile, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import numpy as np

from app.inference import ECGInferenceEngine
from digitizer.pipeline import ECGDigitizerPipeline

app = FastAPI(title="ECG Interpreter API", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

engine: Optional[ECGInferenceEngine] = None
digitizer: Optional[ECGDigitizerPipeline] = None

LEAD_NAMES = ["I", "II", "III", "aVR", "aVL", "aVF", "V1", "V2", "V3", "V4", "V5", "V6"]


@app.on_event("startup")
async def startup_event():
    global engine, digitizer
    try:
        engine = ECGInferenceEngine()
        digitizer = ECGDigitizerPipeline()
        print("[API] Models loaded successfully.")
    except Exception as e:
        print(f"[API] Warning: could not load models at startup: {e}")


@app.get("/health")
async def health():
    return {
        "status": "ok",
        "echonext_loaded": engine is not None and engine.model is not None,
        "ptbxl_loaded": engine is not None and engine.ptbxl_engine is not None,
    }


def _serialize_result(result: dict, waveform: np.ndarray, meta: dict) -> dict:
    """Convert inference result + waveform to a JSON-serialisable response."""
    # Downsample waveform (12, 2500) → (12, 500) for mobile bandwidth
    waveform_display = waveform[:, ::5].tolist()

    # Normalise diagnoses to plain dicts
    raw_diagnoses = result.get("diagnoses") or []
    diagnoses = []
    for d in raw_diagnoses:
        diagnoses.append(
            {
                "name": str(d.get("name", "")),
                "confidence": float(d.get("confidence", 0)),
                "level": str(d.get("level", "POSSIBLE")),
                "description": str(d.get("description", "")),
                "findings": [str(f) for f in (d.get("findings") or [])],
                "source": str(d.get("source", "")),
            }
        )

    # PTB-XL results
    ptbxl = result.get("ptbxl_result")
    ptbxl_serialised = None
    if ptbxl:
        ptbxl_serialised = {
            "probabilities": [float(p) for p in ptbxl.get("probabilities", [])],
            "labels": [str(l) for l in ptbxl.get("labels", [])],
        }

    # ECG features
    ecg_features = result.get("ecg_features") or {}
    safe_features = {}
    for k, v in ecg_features.items():
        try:
            safe_features[str(k)] = float(v) if v is not None else None
        except (TypeError, ValueError):
            safe_features[str(k)] = str(v)

    probs = result.get("probabilities", [])
    shd_prob = float(probs[11]) if len(probs) > 11 else 0.0

    return {
        "success": True,
        "risk_level": str(result.get("risk_level", "LOW")),
        "shd_probability": shd_prob,
        "ecg_features": safe_features,
        "probabilities": [float(p) for p in probs],
        "labels": [str(l) for l in result.get("labels", [])],
        "diagnoses": diagnoses,
        "ptbxl_result": ptbxl_serialised,
        "waveform": waveform_display,
        "lead_names": LEAD_NAMES,
        "waveform_meta": {
            "leads_detected": int(meta.get("leads_detected", 12)),
            "source": str(meta.get("source", "uploaded image")),
        },
    }


@app.post("/analyze")
async def analyze_ecg(
    file: UploadFile = File(...),
    age: Optional[int] = Form(None),
    sex: Optional[str] = Form(None),
):
    if engine is None or digitizer is None:
        raise HTTPException(status_code=503, detail="Models not loaded")

    file_bytes = await file.read()
    if len(file_bytes) > 20 * 1024 * 1024:
        raise HTTPException(status_code=413, detail="File too large (max 20 MB)")

    patient_info: dict = {}
    if age is not None:
        patient_info["age"] = age
    if sex:
        patient_info["sex"] = sex

    try:
        waveform, meta = digitizer.process(file_bytes, filename=file.filename or "upload")
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=422, detail=f"Digitization failed: {e}")

    if waveform is None:
        raise HTTPException(
            status_code=422,
            detail=meta.get("error", "Could not extract ECG waveform from image."),
        )

    try:
        result = engine.predict(waveform, patient_info)
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Inference failed: {e}")

    return _serialize_result(result, waveform, meta)


@app.post("/analyze/demo")
async def analyze_demo():
    """Analyse a built-in PTB-XL sample ECG (no upload required)."""
    if engine is None:
        raise HTTPException(status_code=503, detail="Models not loaded")

    try:
        import wfdb  # type: ignore

        # Find any available PTB-XL record in the data directory
        data_dir = os.path.join(os.path.dirname(__file__), "..", "data")
        record_path = None
        for root, _, files in os.walk(data_dir):
            for f in files:
                if f.endswith(".hea"):
                    record_path = os.path.join(root, f[:-4])
                    break
            if record_path:
                break

        if record_path is None:
            raise HTTPException(status_code=404, detail="No demo ECG records found.")

        record = wfdb.rdrecord(record_path, sampto=2500)
        waveform = record.p_signal.T.astype(np.float32)  # (12, 2500)
        result = engine.predict(waveform, {})
        return _serialize_result(result, waveform, {"source": "PTB-XL demo", "leads_detected": 12})

    except HTTPException:
        raise
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))
