"""
main.py - FastAPI backend + static frontend server.

Serves a model trained on the REAL contracted-duration target
(tender.contractPeriod.durationInDays from OCDS). All predictions are the
*contracted* execution period agreed at signing - not actual completion time.
"""
import json
import os
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

BASE = Path(__file__).parent
MODEL_PATH = BASE / "model.joblib"
META_PATH = BASE / "model_meta.json"
CLEAN_PATH = BASE / "data" / "processed" / "contracts_clean.csv"
FEATURE_META_PATH = BASE / "data" / "processed" / "feature_metadata.json"
STATIC_DIR = BASE.parent / "frontend" / "dist"

CATEGORICAL = ["cpv4", "method", "buyer_type", "season", "postal_region"]
NUMERIC = ["value_log", "num_offers", "month"]

CPV_LABELS = {
    "4520": "site_prep_demolition", "4521": "building_construction",
    "4522": "civil_engineering", "4523": "roads_highways",
    "4524": "water_marine_works", "4525": "other_civil_works",
    "4526": "roof_structural", "4531": "electrical_installation",
    "4532": "insulation_works", "4533": "plumbing_heating",
    "4534": "fencing_railing", "4500": "general_construction",
}
LABEL_TO_CPV = {v: k for k, v in CPV_LABELS.items()}

if not MODEL_PATH.exists():
    raise FileNotFoundError(f"Model not found at {MODEL_PATH}. Run train.py first.")
pipeline = joblib.load(MODEL_PATH)

model_meta = json.loads(META_PATH.read_text(encoding="utf-8")) if META_PATH.exists() else {}
feature_meta = json.loads(FEATURE_META_PATH.read_text(encoding="utf-8")) if FEATURE_META_PATH.exists() else {}
MEAN_DAYS = feature_meta.get("mean_days", 204)

# Cache the repairs table once (static per deploy)
_repairs_df = pd.read_csv(CLEAN_PATH) if CLEAN_PATH.exists() else pd.DataFrame()

app = FastAPI(title="Infrastructure Duration Predictor")

# CORS: restrict to known frontend origin(s). Override with ALLOWED_ORIGINS
# (comma-separated) at deploy time. No wildcard + credentials (invalid combo).
_origins = os.environ.get(
    "ALLOWED_ORIGINS",
    "http://infrastructure.stormlabs.cloud,https://infrastructure.stormlabs.cloud",
).split(",")
app.add_middleware(
    CORSMiddleware,
    allow_origins=[o.strip() for o in _origins if o.strip()],
    allow_credentials=False,
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)


def risk_level(days: float) -> str:
    if days <= MEAN_DAYS * 0.8:
        return "low"
    if days <= MEAN_DAYS * 1.3:
        return "medium"
    return "high"


class PredictRequest(BaseModel):
    repair_type: str = "roads_highways"
    method: str = "open"
    buyer_type: str = "municipality"
    season: str = "summer"
    value_bgn: float = Field(default=150000.0, ge=0)
    num_offers: int = Field(default=3, ge=0)
    month: int = Field(default=6, ge=1, le=12)
    postal_region: str = "??"


class PredictResponse(BaseModel):
    predicted_days: float
    delay_risk: str
    confidence: str
    features_used: dict
    model_info: dict


def features_from(cpv4, method, buyer_type, season, value_bgn, num_offers, month, postal_region):
    return {
        "cpv4": str(cpv4), "method": str(method), "buyer_type": str(buyer_type),
        "season": str(season), "postal_region": str(postal_region or "??"),
        "value_log": float(np.log1p(max(0.0, value_bgn))),
        "num_offers": int(num_offers), "month": int(month),
    }


@app.get("/api/health")
def health():
    return {"status": "healthy", "target": model_meta.get("target"),
            "trained_at": model_meta.get("trained_at"),
            "num_samples": model_meta.get("num_samples"),
            "test_r2": model_meta.get("test_r2"),
            "test_mae": model_meta.get("test_mae"),
            "baseline_mae": model_meta.get("baseline_mae")}


@app.get("/api/meta")
def get_metadata():
    return {
        "repair_types": feature_meta.get("repair_types", []),
        "methods": feature_meta.get("methods", []),
        "object_types": feature_meta.get("object_types", []),
        "seasons": feature_meta.get("seasons", []),
        "authority_types": feature_meta.get("authority_types", []),
        "towns": feature_meta.get("towns", []),
        "mean_days": MEAN_DAYS,
        "num_samples": feature_meta.get("num_samples", 0),
        "target_note": feature_meta.get("target_note", ""),
    }


@app.get("/api/repairs")
def get_repairs(town: str = None):
    if _repairs_df.empty:
        return {"repairs": [], "total": 0, "error": "No data available"}

    df = _repairs_df
    if town and town != "all":
        df = df[df["town"].astype(str).str.lower() == town.lower()]
    if df.empty:
        return {"repairs": [], "total": 0}

    X = df[CATEGORICAL + NUMERIC].copy()
    for c in NUMERIC:
        X[c] = pd.to_numeric(X[c], errors="coerce").fillna(0)
    for c in CATEGORICAL:
        X[c] = X[c].astype(str)
    preds = pipeline.predict(X)

    repairs = []
    for pred, (_, row) in zip(preds, df.iterrows()):
        days = max(1, round(float(pred)))
        repairs.append({
            "id": str(row.get("id", "")),
            "contract_number": str(row.get("contract_number", "")),
            "town": str(row.get("town", "неизвестен")),
            "lat": float(row.get("lat", 0) or 0),
            "lng": float(row.get("lng", 0) or 0),
            "repair_type": str(row.get("repair_type", "")),
            "object_type": str(row.get("object_type", "")),
            "subject": str(row.get("subject", ""))[:200],
            "authority": str(row.get("authority", "")),
            "value_bgn": float(row.get("value_bgn", 0) or 0),
            "actual_contracted_days": int(row.get("contracted_days", 0) or 0),
            "predicted_days": days,
            "delay_risk": risk_level(days),
            "season": str(row.get("season", "")),
        })
    return {"repairs": repairs, "total": len(repairs),
            "model_test_mae": model_meta.get("test_mae"),
            "model_test_r2": model_meta.get("test_r2")}


@app.post("/api/predict", response_model=PredictResponse)
def predict(req: PredictRequest):
    try:
        cpv4 = LABEL_TO_CPV.get(req.repair_type, "4500")
        feats = features_from(cpv4, req.method, req.buyer_type, req.season,
                              req.value_bgn, req.num_offers, req.month, req.postal_region)
        X = pd.DataFrame([feats])
        days = max(1, round(float(pipeline.predict(X)[0])))
        return PredictResponse(
            predicted_days=days,
            delay_risk=risk_level(days),
            confidence="medium",
            features_used=feats,
            model_info={
                "target": model_meta.get("target"),
                "target_note": model_meta.get("target_note"),
                "test_mae": model_meta.get("test_mae"),
                "test_r2": model_meta.get("test_r2"),
                "baseline_mae": model_meta.get("baseline_mae"),
                "trained_at": model_meta.get("trained_at"),
            },
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


if STATIC_DIR.exists():
    app.mount("/", StaticFiles(directory=str(STATIC_DIR), html=True), name="static")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=3000, reload=False)
