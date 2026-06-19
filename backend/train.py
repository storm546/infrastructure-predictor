"""
train.py - Train XGBoost model to predict infrastructure repair duration.

Features:
  - repair_type (one-hot)
  - object_type (one-hot)
  - season (one-hot)
  - authority_type (one-hot)
  - value_log, num_offers, eu_financed, month, day_of_week
  - has_annex, annex_extension_days

Target: actual_days (estimated completion duration)
"""

import pandas as pd
import numpy as np
import joblib
import json
from pathlib import Path
from datetime import datetime

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import xgboost as xgb

DATA_DIR = Path(__file__).parent / "data"
PROCESSED_DIR = DATA_DIR / "processed"
MODEL_PATH = Path(__file__).parent / "model.joblib"
ENCODER_PATH = Path(__file__).parent / "encoder.joblib"


def build_pipeline(categorical_cols: list[str], numeric_cols: list[str]) -> Pipeline:
    """Build a ColumnTransformer + XGBoost pipeline."""
    preprocessor = ColumnTransformer(
        transformers=[
            ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), categorical_cols),
            ("num", StandardScaler(), numeric_cols),
        ],
        remainder="drop",
    )

    model = xgb.XGBRegressor(
        n_estimators=200,
        max_depth=6,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        min_child_weight=3,
        reg_alpha=0.1,
        reg_lambda=1.0,
        random_state=42,
        n_jobs=-1,
        objective="reg:squarederror",
        eval_metric="mae",
    )

    pipeline = Pipeline([
        ("preprocessor", preprocessor),
        ("regressor", model),
    ])

    return pipeline


def evaluate_model(pipeline: Pipeline, X_train, X_test, y_train, y_test):
    """Print comprehensive evaluation metrics."""
    y_pred_train = pipeline.predict(X_train)
    y_pred_test = pipeline.predict(X_test)

    train_mae = mean_absolute_error(y_train, y_pred_train)
    test_mae = mean_absolute_error(y_test, y_pred_test)
    train_rmse = np.sqrt(mean_squared_error(y_train, y_pred_train))
    test_rmse = np.sqrt(mean_squared_error(y_test, y_pred_test))
    train_r2 = r2_score(y_train, y_pred_train)
    test_r2 = r2_score(y_test, y_pred_test)

    print(f"\n{'=' * 60}")
    print(f"  MODEL EVALUATION")
    print(f"{'=' * 60}")
    print(f"  {'':>12} {'Train':>12} {'Test':>12}")
    print(f"  {'MAE':>12} {train_mae:>12.1f} {test_mae:>12.1f}")
    print(f"  {'RMSE':>12} {train_rmse:>12.1f} {test_rmse:>12.1f}")
    print(f"  {'R²':>12} {train_r2:>12.3f} {test_r2:>12.3f}")
    print(f"{'=' * 60}")

    # Feature importance
    try:
        # Get feature names after one-hot encoding
        preprocessor = pipeline.named_steps["preprocessor"]
        ohe = preprocessor.named_transformers_["cat"]
        cat_features = ohe.get_feature_names_out(preprocessor.transformers_[0][2])

        feature_names = list(cat_features) + preprocessor.transformers_[1][2]
        importances = pipeline.named_steps["regressor"].feature_importances_

        # Ensure lengths match
        if len(feature_names) == len(importances):
            feat_imp = sorted(zip(feature_names, importances), key=lambda x: x[1], reverse=True)
            print(f"\n  Top 10 features:")
            for name, imp in feat_imp[:10]:
                print(f"    {name:40s} {imp:.4f}")
    except Exception as e:
        print(f"  (Feature importance unavailable: {e})")

    # Cross-validation
    print(f"\n  5-fold CV MAE scores:")
    try:
        cv_scores = cross_val_score(
            pipeline, pd.concat([X_train, X_test]), pd.concat([y_train, y_test]),
            cv=5, scoring="neg_mean_absolute_error", n_jobs=-1
        )
        print(f"    Individual: {[-round(s, 1) for s in cv_scores]}")
        print(f"    Mean CV MAE: {-cv_scores.mean():.1f} (±{cv_scores.std():.1f})")
    except Exception as e:
        print(f"    (CV failed: {e})")

    return test_mae, test_r2


def main():
    print("=" * 60)
    print("  XGBoost Model Training")
    print("=" * 60)

    # Load cleaned data
    clean_path = PROCESSED_DIR / "contracts_clean.csv"
    if not clean_path.exists():
        print(f"ERROR: {clean_path} not found. Run clean_data.py first.")
        return

    df = pd.read_csv(clean_path)
    print(f"\nLoaded {len(df)} records")

    # Define feature columns
    categorical_cols = ["repair_type", "object_type", "season", "authority_type"]
    numeric_cols = [
        "value_log", "value_bgn_normalized", "num_offers",
        "month", "day_of_week", "has_annex", "annex_extension_days",
        "eu_financed",
    ]

    # Ensure all columns exist
    available_cols = [c for c in categorical_cols + numeric_cols if c in df.columns]
    categorical_cols = [c for c in categorical_cols if c in df.columns]
    numeric_cols = [c for c in numeric_cols if c in df.columns]

    print(f"  Categorical features: {categorical_cols}")
    print(f"  Numeric features: {numeric_cols}")

    # Prepare X and y
    target_col = "actual_days"
    y = df[target_col].values
    X = df[available_cols]

    # Handle any remaining NaNs
    X = X.fillna(0)

    # Train/test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    print(f"\n  Train: {len(X_train)} samples")
    print(f"  Test:  {len(X_test)} samples")
    print(f"  Target range: {y.min():.0f} - {y.max():.0f} days (mean: {y.mean():.0f})")

    # Build and train
    print(f"\nTraining XGBoost...")
    pipeline = build_pipeline(categorical_cols, numeric_cols)
    pipeline.fit(X_train, y_train)

    # Evaluate
    test_mae, test_r2 = evaluate_model(pipeline, X_train, X_test, y_train, y_test)

    # Save model
    joblib.dump(pipeline, MODEL_PATH)
    print(f"\n  Model saved → {MODEL_PATH}")

    # Save column metadata for API
    meta = {
        "categorical_cols": categorical_cols,
        "numeric_cols": numeric_cols,
        "target": "actual_days",
        "test_mae": float(test_mae),
        "test_r2": float(test_r2),
        "trained_at": datetime.now().isoformat(),
        "num_samples": len(df),
    }
    meta_path = Path(__file__).parent / "model_meta.json"
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)
    print(f"  Metadata saved → {meta_path}")


if __name__ == "__main__":
    main()
