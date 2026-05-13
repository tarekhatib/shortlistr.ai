import os
import sys
import tempfile
from contextlib import asynccontextmanager
from dotenv import load_dotenv
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware

load_dotenv(os.path.join(os.path.dirname(__file__), "..", ".env"))

sys.path.insert(0, os.path.dirname(__file__))
from gemini_extractor import extract_features_from_cv
from predict import (
    encode_candidate,
    generate_feedback,
    load_models,
    scale_candidate,
)

models: dict = {}

@asynccontextmanager
async def lifespan(app: FastAPI):
    print("Loading models…")
    rf, mlp, scaler, benchmarks = load_models()
    models["rf"] = rf
    models["mlp"] = mlp
    models["scaler"] = scaler
    models["benchmarks"] = benchmarks
    print("Models ready.")
    yield


app = FastAPI(title="Resume Screener API", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173"],
    allow_methods=["POST", "GET"],
    allow_headers=["*"],
)

@app.get("/health")
def health():
    return {"status": "ok", "models_loaded": bool(models)}

@app.post("/analyze")
async def analyze(cv: UploadFile = File(...), jd: str = Form(...)):
    if not cv.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are accepted.")
    if not jd.strip():
        raise HTTPException(status_code=400, detail="Job description cannot be empty.")

    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as f:
        f.write(await cv.read())
        pdf_path = f.name

    with tempfile.NamedTemporaryFile(delete=False, suffix=".txt", mode="w", encoding="utf-8") as f:
        f.write(jd)
        jd_path = f.name

    try:
        gemini_result = extract_features_from_cv(pdf_path, jd_path)
    except Exception as e:
        raise HTTPException(status_code=503, detail=str(e))
    finally:
        os.unlink(pdf_path)
        os.unlink(jd_path)

    job_requirements = gemini_result.pop("job_requirements", {}) or {}

    features_enc = encode_candidate(gemini_result)
    row_scaled = scale_candidate(features_enc, models["scaler"])

    rf_pred = int(models["rf"].predict(row_scaled)[0])
    rf_prob = round(float(models["rf"].predict_proba(row_scaled)[0][rf_pred]) * 100, 1)

    mlp_prob_raw = float(models["mlp"].predict(row_scaled, verbose=0)[0][0])
    mlp_pred = 1 if mlp_prob_raw >= 0.5 else 0
    mlp_prob = round((mlp_prob_raw if mlp_pred == 1 else (1 - mlp_prob_raw)) * 100, 1)

    if rf_pred == mlp_pred:
        final_pred = rf_pred
    else:
        final_pred = rf_pred if rf_prob >= mlp_prob else mlp_pred

    feedback = generate_feedback(features_enc, models["rf"], models["benchmarks"], job_requirements) if final_pred == 0 else []

    return {
        "features": gemini_result,
        "verdict": "shortlisted" if final_pred == 1 else "not_shortlisted",
        "predictions": {
            "random_forest": {"verdict": "shortlisted" if rf_pred == 1 else "not_shortlisted", "confidence": rf_prob},
            "neural_network": {"verdict": "shortlisted" if mlp_pred == 1 else "not_shortlisted", "confidence": mlp_prob},
        },
        "feedback": feedback,
    }
