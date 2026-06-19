"""
train.py - Train an XGBoost model on the REAL contracted-duration target.

Target: contracted_days  (tender.contractPeriod.durationInDays from OCDS)

Unlike the previous version, this script:
  - trains on a real, published target (not a hash-seeded RNG),
  - reports a mean-predictor BASELINE so the metrics are interpretable,
  - persists BOTH train and test R2 so the over/under-fit gap is visible.
"""

import json
import math
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

BASE = Path(__file__).parent
PROCESSED = BASE / "data" / "processed" / "contracts_clean.csv"
MODEL_PATH = BASE / "model.joblib"
META_PATH = BASE / "model_meta.json"

CATEGORICAL = ["cpv4", "method", "buyer_type", "season", "postal_region"]
NUMERIC = ["value_log", "num_offers", "month"]
TARGET = "contracted_days"


def build_pipeline() -> Pipeline:
    pre = ColumnTransformer([
        ("cat", OneHotEncoder(handle_unknown="ignore"), CATEGORICAL),
        ("num", StandardScaler(), NUMERIC),
    ])
    model = xgb.XGBRegressor(
        n_estimators=400, max_depth=5, learning_rate=0.03,
        subsample=0.8, colsample_bytree=0.8, min_child_weight=5,
        reg_alpha=0.1, reg_lambda=1.5, random_state=42, n_jobs=-1,
        objective="reg:squarederror", eval_metric="mae",
    )
    return Pipeline([("pre", pre), ("reg", model)])


def main():
    print("=" * 60)
    print("  XGBoost Training (real contracted-duration target)")
    print("=" * 60)

    if not PROCESSED.exists():
        print(f"ERROR: {PROCESSED} not found. Run clean_data.py first.")
        return

    df = pd.read_csv(PROCESSED)
    df = df.dropna(subset=[TARGET])
    print(f"\n  Loaded {len(df)} construction contracts")
    print(f"  Target mean={df[TARGET].mean():.0f}  median={df[TARGET].median():.0f}  "
          f"std={df[TARGET].std():.0f}")

    X = df[CATEGORICAL + NUMERIC].copy()
    for c in NUMERIC:
        X[c] = pd.to_numeric(X[c], errors="coerce").fillna(0)
    for c in CATEGORICAL:
        X[c] = X[c].astype(str)
    y = df[TARGET].values

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    print(f"  Train: {len(X_train)}   Test: {len(X_test)}")

    pipe = build_pipeline()
    pipe.fit(X_train, y_train)

    def metrics(yt, yp):
        return (mean_absolute_error(yt, yp),
                math.sqrt(mean_squared_error(yt, yp)),
                r2_score(yt, yp))

    tr_mae, tr_rmse, tr_r2 = metrics(y_train, pipe.predict(X_train))
    te_mae, te_rmse, te_r2 = metrics(y_test, pipe.predict(X_test))

    # Baseline: predict the training mean
    dummy = DummyRegressor(strategy="mean").fit(X_train, y_train)
    base_mae, base_rmse, base_r2 = metrics(y_test, dummy.predict(X_test))
    improvement = 100 * (1 - te_mae / base_mae)

    cv = cross_val_score(pipe, X, y, cv=5,
                         scoring="neg_mean_absolute_error", n_jobs=-1)

    print(f"\n  {'':8} {'MAE':>8} {'RMSE':>8} {'R2':>8}")
    print(f"  {'train':8} {tr_mae:8.1f} {tr_rmse:8.1f} {tr_r2:8.3f}")
    print(f"  {'test':8} {te_mae:8.1f} {te_rmse:8.1f} {te_r2:8.3f}")
    print(f"  {'baseline':8} {base_mae:8.1f} {base_rmse:8.1f} {base_r2:8.3f}")
    print(f"\n  MAE improvement over baseline: {improvement:.0f}%")
    print(f"  5-fold CV MAE: {-cv.mean():.1f} (+/- {cv.std():.1f})")

    joblib.dump(pipe, MODEL_PATH)
    print(f"\n  Model saved -> {MODEL_PATH}")

    meta = {
        "target": TARGET,
        "target_note": "Real contracted execution period (durationInDays) from OCDS; "
                       "duration agreed at signing, not actual completion time.",
        "data_source": "Open Contracting / DIGIWHIST Bulgaria (opentender.eu), CC BY-NC-SA 4.0",
        "categorical_cols": CATEGORICAL,
        "numeric_cols": NUMERIC,
        "num_samples": int(len(df)),
        "train_mae": float(tr_mae), "train_r2": float(tr_r2),
        "test_mae": float(te_mae), "test_rmse": float(te_rmse), "test_r2": float(te_r2),
        "baseline_mae": float(base_mae),
        "mae_improvement_pct": float(improvement),
        "cv_mae_mean": float(-cv.mean()), "cv_mae_std": float(cv.std()),
        "trained_at": datetime.now(timezone.utc).isoformat(),
    }
    with open(META_PATH, "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)
    print(f"  Metadata saved -> {META_PATH}")


if __name__ == "__main__":
    main()
