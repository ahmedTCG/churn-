from pathlib import Path
import json
import numpy as np
import pandas as pd

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import SGDClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import roc_auc_score, average_precision_score

import warnings
warnings.filterwarnings("ignore", category=RuntimeWarning, message=".*matmul.*")
warnings.filterwarnings("ignore", category=RuntimeWarning, message=".*raw_prediction.*")
warnings.filterwarnings("ignore", category=RuntimeWarning, message=".*grad.*matmul.*")

PROJECT_ROOT = Path(__file__).resolve().parents[2]
TARGET = "churn_label"
ID_COL = "external_customerkey"
DROP_FEATURES = ["total_revenue", "avg_order_value"]

def prep_X(df: pd.DataFrame) -> pd.DataFrame:
    X = df.drop(columns=[c for c in [ID_COL, TARGET, "snapshot_time"] if c in df.columns]).copy()
    X = X.drop(columns=DROP_FEATURES, errors="ignore")
    X = X.replace([np.inf, -np.inf], np.nan).fillna(0)
    X = X.select_dtypes(include=[np.number])

    # Mirror training stabilizations
    if "recency_days" in X.columns:
        X["recency_days"] = X["recency_days"].clip(upper=365)

    count_like_prefixes = ("n_events_last_", "active_days_last_", "cnt_")
    for c in list(X.columns):
        if c.startswith(count_like_prefixes) or c in {"n_orders"}:
            X[c] = np.log1p(X[c])

    return X

def main():
    data_path = PROJECT_ROOT / "data" / "processed" / "model_dataset_timesplit_30d.parquet"
    df = pd.read_parquet(data_path)

    df["snapshot_time"] = pd.to_datetime(df["snapshot_time"])
    months = sorted(df["snapshot_time"].unique())
    if len(months) < 4:
        raise SystemExit("Need at least 4 snapshots for a meaningful time split.")

    test_months = months[-2:]
    train_months = months[:-2]

    train_df = df[df["snapshot_time"].isin(train_months)].copy()
    test_df = df[df["snapshot_time"].isin(test_months)].copy()

    y_train = train_df[TARGET].astype(int)
    y_test = test_df[TARGET].astype(int)

    X_train = prep_X(train_df)
    X_test = prep_X(test_df)

    # Train model (same as 04_train_logreg_v1.py)
    base = Pipeline(steps=[
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
    base.fit(X_train, y_train)

    calibrated = CalibratedClassifierCV(estimator=base, method="sigmoid", cv=3)
    calibrated.fit(X_train, y_train)

    proba = calibrated.predict_proba(X_test)[:, 1]
    roc = roc_auc_score(y_test, proba)
    pr = average_precision_score(y_test, proba)

    print("Train months:", [str(pd.to_datetime(m).date()) for m in train_months])
    print("Test months :", [str(pd.to_datetime(m).date()) for m in test_months])
    print("Train rows:", len(train_df), "Test rows:", len(test_df))
    print("Test ROC_AUC:", roc)
    print("Test PR_AUC :", pr)

    # Per-month drift view
    print("\nPer-month test metrics:")
    rows = []
    for m in test_months:
        sub = test_df[test_df["snapshot_time"] == m]
        y_m = sub[TARGET].astype(int)
        X_m = prep_X(sub)
        p_m = calibrated.predict_proba(X_m)[:, 1]
        rows.append({
            "snapshot_time": str(pd.to_datetime(m).date()),
            "rows": int(len(sub)),
            "label_mean": float(y_m.mean()),
            "roc_auc": float(roc_auc_score(y_m, p_m)),
            "pr_auc": float(average_precision_score(y_m, p_m)),
        })

    out = pd.DataFrame(rows)
    print(out)

    out_path = PROJECT_ROOT / "artifacts" / "eval_timesplit_30d.csv"
    out.to_csv(out_path, index=False)
    print("\nWrote:", out_path)

if __name__ == "__main__":
    main()
