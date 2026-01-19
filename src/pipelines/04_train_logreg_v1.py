from pathlib import Path
import json
import joblib
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.calibration import CalibratedClassifierCV
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import roc_auc_score, average_precision_score

import warnings

# Silence known numeric-backend RuntimeWarnings during sklearn internal matmul.
# We already verified X has no NaN/Inf and reasonable magnitudes.
warnings.filterwarnings("ignore", category=RuntimeWarning, message=".*matmul.*")
warnings.filterwarnings("ignore", category=RuntimeWarning, message=".*raw_prediction.*")
warnings.filterwarnings("ignore", category=RuntimeWarning, message=".*grad.*matmul.*")

PROJECT_ROOT = Path(__file__).resolve().parents[2]

TARGET = "churn_label"
ID_COL = "external_customerkey"

DROP_FEATURES = ["total_revenue", "avg_order_value"]

def main():
    dataset_path = PROJECT_ROOT / "data" / "processed" / "model_dataset_label_30d.parquet"
    df = pd.read_parquet(dataset_path)

    y = df[TARGET].astype(int)

    X = df.drop(columns=[c for c in [ID_COL, TARGET] if c in df.columns]).copy()
    X = X.drop(columns=DROP_FEATURES, errors="ignore")

    X = X.replace([np.inf, -np.inf], np.nan).fillna(0)
    X = X.select_dtypes(include=[np.number])

    # Drop zero-variance columns (can cause numerical warnings)
    zero_std_cols = X.columns[X.std(axis=0) == 0].tolist()
    if zero_std_cols:
        X = X.drop(columns=zero_std_cols)
        print("Dropped zero-std cols:", zero_std_cols)

    # --------- Stabilize scales (prevents overflow warnings) ----------
    # Cap recency (customers with no strong history get 999 earlier)
    if "recency_days" in X.columns:
        X["recency_days"] = X["recency_days"].clip(upper=365)

    # Log-transform heavy-tailed count features
    count_like_prefixes = ("n_events_last_", "active_days_last_", "cnt_")
    for c in list(X.columns):
        if c.startswith(count_like_prefixes) or c in {"n_orders"}:
            X[c] = np.log1p(X[c])

    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )

    model = Pipeline(steps=[
        ("scaler", StandardScaler()),
        ("clf", SGDClassifier(
            loss="log_loss",
            penalty="l2",
            alpha=1e-4,
            max_iter=5000,
            tol=1e-3,
            class_weight="balanced",
            random_state=42
        ))
    ])

    model.fit(X_train, y_train)

    # -------- Probability calibration --------
    calibrated = CalibratedClassifierCV(
        estimator=model,
        method="sigmoid",
        cv=3
    )
    calibrated.fit(X_train, y_train)

    y_proba = calibrated.predict_proba(X_test)[:, 1]
    assert np.isfinite(y_proba).all(), 'Non-finite probabilities produced'
    roc = roc_auc_score(y_test, y_proba)
    pr = average_precision_score(y_test, y_proba)

    artifacts = PROJECT_ROOT / "artifacts"
    artifacts.mkdir(exist_ok=True)

    model_path = artifacts / "churn_model_v1.joblib"
    joblib.dump(calibrated, model_path)

    feature_list = list(X_train.columns)
    (artifacts / "feature_list.json").write_text(json.dumps(feature_list, indent=2), encoding="utf-8")

    # Save simple training metadata
    meta = {
        "dataset": str(dataset_path),
        "n_rows": int(len(df)),
        "n_features": int(len(feature_list)),
        "churn_rate": float(y.mean()),
        "roc_auc": float(roc),
        "pr_auc": float(pr),
        "drop_features": DROP_FEATURES,
        "random_state": 42,
    }
    (artifacts / "train_meta_v1.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")

    print("Saved:", model_path)
    print("ROC_AUC:", roc)
    print("PR_AUC:", pr)
    print("Meta:", artifacts / "train_meta_v1.json")

if __name__ == "__main__":
    main()
