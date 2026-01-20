from pathlib import Path
import json
import joblib
import numpy as np
import pandas as pd

from sklearn.pipeline import Pipeline
from sklearn.calibration import CalibratedClassifierCV
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import roc_auc_score, average_precision_score, classification_report

import warnings
warnings.filterwarnings("ignore", category=RuntimeWarning, message=".*matmul.*")
warnings.filterwarnings("ignore", category=RuntimeWarning, message=".*raw_prediction.*")
warnings.filterwarnings("ignore", category=RuntimeWarning, message=".*grad.*matmul.*")

PROJECT_ROOT = Path(__file__).resolve().parents[2]

TARGET = "churn_label"
ID_COL = "external_customerkey"

# keep consistent with your current trainer
DROP_FEATURES = ["total_revenue", "avg_order_value"]

# time split: last 20% of timeline as test
TEST_FRACTION = 0.20


def main():
    dataset_path = PROJECT_ROOT / "data" / "processed" / "model_dataset_label_30d.parquet"
    df = pd.read_parquet(dataset_path).copy()

    # Ensure we have a time column (from features pipeline)
    if "snapshot_time" in df.columns:
        tcol = "snapshot_time"
    elif "event_time" in df.columns:
        tcol = "event_time"
    else:
        raise ValueError("No time column found. Add snapshot_time to dataset or pass event_time through features.")

    df[tcol] = pd.to_datetime(df[tcol], errors="coerce")
    df = df.dropna(subset=[tcol, TARGET]).copy()

    # Sort by time and split
    df = df.sort_values(tcol).reset_index(drop=True)
    split_idx = int(len(df) * (1 - TEST_FRACTION))

    train_df = df.iloc[:split_idx].copy()
    test_df  = df.iloc[split_idx:].copy()

    y_train = train_df[TARGET].astype(int)
    y_test  = test_df[TARGET].astype(int)

    X_train = train_df.drop(columns=[c for c in [ID_COL, TARGET] if c in train_df.columns], errors="ignore")
    X_test  = test_df.drop(columns=[c for c in [ID_COL, TARGET] if c in test_df.columns], errors="ignore")

    X_train = X_train.drop(columns=DROP_FEATURES, errors="ignore")
    X_test  = X_test.drop(columns=DROP_FEATURES, errors="ignore")

    X_train = X_train.replace([np.inf, -np.inf], np.nan).fillna(0).select_dtypes(include=[np.number])
    X_test  = X_test.replace([np.inf, -np.inf], np.nan).fillna(0).select_dtypes(include=[np.number])

    # Align columns exactly (train defines schema)
    X_test = X_test.reindex(columns=X_train.columns, fill_value=0)

    # Safety: drop zero-variance
    zero_std_cols = X_train.columns[X_train.std(axis=0) == 0].tolist()
    if zero_std_cols:
        X_train = X_train.drop(columns=zero_std_cols)
        X_test  = X_test.drop(columns=zero_std_cols)
        print("Dropped zero-std cols:", zero_std_cols)

    # Same scaling + classifier as your v1
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

    calibrated = CalibratedClassifierCV(
        estimator=model,
        method="sigmoid",
        cv=3
    )
    calibrated.fit(X_train, y_train)

    y_proba = calibrated.predict_proba(X_test)[:, 1]
    roc = roc_auc_score(y_test, y_proba)
    pr = average_precision_score(y_test, y_proba)

    y_pred = (y_proba >= 0.5).astype(int)

    artifacts = PROJECT_ROOT / "artifacts"
    artifacts.mkdir(exist_ok=True)

    model_path = artifacts / "churn_model_timesplit_v1.joblib"
    joblib.dump(calibrated, model_path)

    feature_list = list(X_train.columns)
    (artifacts / "feature_list_timesplit.json").write_text(json.dumps(feature_list, indent=2), encoding="utf-8")

    meta = {
        "dataset": str(dataset_path),
        "time_col": tcol,
        "test_fraction": TEST_FRACTION,
        "train_rows": int(len(train_df)),
        "test_rows": int(len(test_df)),
        "train_churn_rate": float(y_train.mean()),
        "test_churn_rate": float(y_test.mean()),
        "roc_auc": float(roc),
        "pr_auc": float(pr),
        "drop_features": DROP_FEATURES,
        "random_state": 42,
    }
    (artifacts / "train_meta_timesplit_v1.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")

    print("Saved:", model_path)
    print("ROC_AUC:", roc)
    print("PR_AUC:", pr)
    print("\nClassification report @0.5:\n")
    print(classification_report(y_test, y_pred, digits=4))
    print("Meta:", artifacts / "train_meta_timesplit_v1.json")


if __name__ == "__main__":
    main()
