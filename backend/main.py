"""
main.py - FastAPI backend + static frontend server.

Serves:
  /          -> static frontend (Vite build)
  /api/meta  -> feature metadata for UI dropdowns
  /api/predict -> predict repair duration
"""
import json
from pathlib import Path

import joblib
import pandas as pd
import numpy as np

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

# --- Config ---
MODEL_PATH = Path(__file__).parent / "model.joblib"
META_PATH = Path(__file__).parent / "model_meta.json"
FEATURE_META_PATH = Path(__file__).parent / "data" / "processed" / "feature_metadata.json"
STATIC_DIR = Path(__file__).parent.parent / "frontend" / "dist"

# --- Load model ---
if not MODEL_PATH.exists():
    raise FileNotFoundError(f"Model not found at {MODEL_PATH}")
pipeline = joblib.load(MODEL_PATH)

# --- Load metadata ---
model_meta = {}
if META_PATH.exists():
    with open(META_PATH) as f:
        model_meta = json.load(f)

feature_meta = {}
if FEATURE_META_PATH.exists():
    with open(FEATURE_META_PATH) as f:
        feature_meta = json.load(f)

# --- App ---
app = FastAPI(title="Infrastructure Repair Predictor")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# --- Schemas ---
class PredictRequest(BaseModel):
    repair_type: str = "road_repair"
    object_type: str = "строителство"
    season: str = "summer"
    authority_type: str = "municipality"
    value_bgn: float = Field(default=150000.0, ge=0)
    num_offers: int = Field(default=3, ge=0)
    eu_financed: int = Field(default=0, ge=0, le=1)
    month: int = Field(default=6, ge=1, le=12)
    day_of_week: int = Field(default=2, ge=0, le=6)
    has_annex: int = Field(default=0, ge=0, le=1)
    annex_extension_days: int = Field(default=0, ge=0)


class PredictResponse(BaseModel):
    predicted_days: float
    delay_risk: str
    confidence: str
    features_used: dict
    model_info: dict


# --- API Routes ---
@app.get("/api/health")
def health():
    return {
        "status": "healthy",
        "model": model_meta.get("target"),
        "trained_at": model_meta.get("trained_at"),
        "num_samples": model_meta.get("num_samples"),
    }


@app.get("/api/meta")
def get_metadata():
    return {
        "repair_types": feature_meta.get("repair_types", []),
        "contractors": feature_meta.get("contractors", []),
        "object_types": feature_meta.get("object_types", []),
        "seasons": feature_meta.get("seasons", []),
        "authority_types": feature_meta.get("authority_types", []),
        "towns": feature_meta.get("towns", []),
        "mean_days": feature_meta.get("mean_days", 0),
        "num_samples": feature_meta.get("num_samples", 0),
    }


@app.get("/api/repairs")
def get_repairs(town: str = None):
    """Return all active repairs with geo data and model predictions."""
    clean_path = Path(__file__).parent / "data" / "processed" / "contracts_clean.csv"
    if not clean_path.exists():
        return {"repairs": [], "error": "No data available"}

    df = pd.read_csv(clean_path)
    
    if town and town != "all":
        df = df[df["town"].str.lower() == town.lower()]
    
    # Build feature matrix for batch prediction
    feature_rows = []
    for _, row in df.iterrows():
        feature_rows.append({
            "repair_type": str(row.get("repair_type", "other")),
            "object_type": str(row.get("object_type", "строителство")),
            "season": str(row.get("season", "summer")),
            "authority_type": str(row.get("authority_type", "municipality")),
            "value_log": np.log1p(float(row.get("value_bgn", 0))),
            "value_bgn_normalized": float(row.get("value_bgn", 0)),
            "num_offers": int(row.get("num_offers", 0)),
            "month": int(row.get("month", 6)),
            "day_of_week": int(row.get("day_of_week", 2)),
            "has_annex": int(row.get("has_annex", 0)),
            "annex_extension_days": int(row.get("annex_extension_days", 0)),
            "eu_financed": int(row.get("eu_financed", 0)),
        })
    
    X_batch = pd.DataFrame(feature_rows)
    predictions = pipeline.predict(X_batch)
    
    mean_days = feature_meta.get("mean_days", 257)
    repairs = []
    for i, (_, row) in enumerate(df.iterrows()):
        pred_days = max(1, round(float(predictions[i])))
        if pred_days <= mean_days * 0.8:
            risk = "low"
        elif pred_days <= mean_days * 1.3:
            risk = "medium"
        else:
            risk = "high"
        
        repairs.append({
            "id": str(row.get("ID на поръчката", "")),
            "contract_number": str(row.get("ДОГОВОР НОМЕР", "")),
            "town": str(row.get("town", "неизвестен")),
            "lat": float(row.get("lat", 0)),
            "lng": float(row.get("lng", 0)),
            "repair_type": str(row.get("repair_type", "")),
            "object_type": str(row.get("object_type", "")),
            "subject": str(row.get("ПРЕДМЕТ на договора", ""))[:200],
            "contractor": str(row.get("contractor", "")),
            "value_bgn": float(row.get("value_bgn", 0)),
            "predicted_days": pred_days,
            "delay_risk": risk,
            "season": str(row.get("season", "")),
            "authority": str(row.get("authority", "")),
            "date": str(row.get("contract_date_parsed", ""))[:10],
        })
    
    return {"repairs": repairs, "total": len(repairs), "model_mae": model_meta.get("test_mae")}


@app.post("/api/predict", response_model=PredictResponse)
def predict(req: PredictRequest):
    try:
        input_data = {
            "repair_type": req.repair_type,
            "object_type": req.object_type,
            "season": req.season,
            "authority_type": req.authority_type,
            "value_log": np.log1p(req.value_bgn),
            "value_bgn_normalized": req.value_bgn,
            "num_offers": req.num_offers,
            "month": req.month,
            "day_of_week": req.day_of_week,
            "has_annex": req.has_annex,
            "annex_extension_days": req.annex_extension_days,
            "eu_financed": req.eu_financed,
        }
        X = pd.DataFrame([input_data])
        prediction = float(pipeline.predict(X)[0])
        predicted_days = max(1, round(prediction))

        mean_days = feature_meta.get("mean_days", 257)
        if predicted_days <= mean_days * 0.8:
            delay_risk = "low"
        elif predicted_days <= mean_days * 1.3:
            delay_risk = "medium"
        else:
            delay_risk = "high"

        return PredictResponse(
            predicted_days=predicted_days,
            delay_risk=delay_risk,
            confidence="low" if model_meta.get("num_samples", 0) < 500 else "medium",
            features_used=input_data,
            model_info={
                "test_mae": model_meta.get("test_mae"),
                "test_r2": model_meta.get("test_r2"),
                "trained_at": model_meta.get("trained_at"),
            },
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# Mount static frontend AFTER api routes (so /api/* takes priority)
if STATIC_DIR.exists():
    app.mount("/", StaticFiles(directory=str(STATIC_DIR), html=True), name="static")

# --- Run ---
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=3000, reload=False)
