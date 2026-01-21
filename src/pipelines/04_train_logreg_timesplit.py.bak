from pathlib import Path
import json
import warnings

import numpy as np
import pandas as pd
import joblib

from sklearn.pipeline import Pipeline
from sklearn.calibration import CalibratedClassifierCV
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import roc_auc_score, average_precision_score, classification_report, confusion_matrix

from src.features.build_features import build_customer_features, filter_strong_events

warnings.filterwarnings("ignore", category=RuntimeWarning, message=".*matmul.*")
warnings.filterwarnings("ignore", category=RuntimeWarning, message=".*raw_prediction.*")
warnings.filterwarnings("ignore", category=RuntimeWarning, message=".*grad.*matmul.*")

PROJECT_ROOT = Path(__file__).resolve().parents[2]
ID_COL = "external_customerkey"
TARGET = "churn_label"
DROP_FEATURES = ["total_revenue", "avg_order_value"]
CHURN_WINDOW_DAYS = 30


def make_dataset(df: pd.DataFrame, snapshot_time: pd.Timestamp) -> pd.DataFrame:
    # labels
    future_end = snapshot_time + pd.Timedelta(days=CHURN_WINDOW_DAYS)
    df_hist = df[df["event_time"] <= snapshot_time]
    customers = df_hist[[ID_COL]].drop_duplicates()

    df_strong = filter_strong_events(df)
    active = df_strong[
        (df_strong["event_time"] > snapshot_time) &
        (df_strong["event_time"] <= future_end)
    ][ID_COL].drop_duplicates()

    labels = customers.copy()
    labels[TARGET] = (~labels[ID_COL].isin(active)).astype(int)

    # features (history only)
    feats = build_customer_features(df, snapshot_time=snapshot_time, churn_window_days=CHURN_WINDOW_DAYS)

    out = feats.merge(labels, on=ID_COL, how="inner")
    out = out.replace([np.inf, -np.inf], np.nan)
    num = out.select_dtypes(include=[np.number]).columns
    out[num] = out[num].fillna(0)
    return out


def make_X(ds: pd.DataFrame) -> pd.DataFrame:
    X = ds.drop(columns=[ID_COL, TARGET], errors="ignore")
    X = X.drop(columns=DROP_FEATURES, errors="ignore")
    X = X.select_dtypes(include=[np.number]).replace([np.inf, -np.inf], np.nan).fillna(0)

    # basic stability
    if "recency_days" in X.columns:
        X["recency_days"] = X["recency_days"].clip(upper=365)

    for c in list(X.columns):
        if c.startswith(("n_events_last_", "active_days_last_", "cnt_")) or c == "n_orders":
            X[c] = np.log1p(X[c])

    # drop constant cols
    X = X.loc[:, X.std(axis=0) != 0]
    return X


def main():
    df = pd.read_parquet(PROJECT_ROOT / "data" / "processed" / "interactions.parquet")
    df["event_time"] = pd.to_datetime(df["event_time"], errors="coerce")
    df = df.dropna(subset=[ID_COL, "event_time", "interaction_type"]).copy()

    max_time = df["event_time"].max()
    snap_test = max_time - pd.Timedelta(days=CHURN_WINDOW_DAYS)
    snap_train = max_time - pd.Timedelta(days=2 * CHURN_WINDOW_DAYS)

    train_ds = make_dataset(df, snap_train)
    test_ds = make_dataset(df, snap_test)

    y_train = train_ds[TARGET].astype(int)
    y_test = test_ds[TARGET].astype(int)

    X_train = make_X(train_ds)
    X_test = make_X(test_ds)

    # align
    for c in X_train.columns:
        if c not in X_test.columns:
            X_test[c] = 0
    X_test = X_test[X_train.columns]

    base = Pipeline([
        ("scaler", StandardScaler()),
        ("clf", SGDClassifier(
            loss="log_loss",
            penalty="l2",
            alpha=1e-4,
            max_iter=5000,
            tol=1e-3,
            class_weight="balanced",
            random_state=42,
        )),
    ])
    base.fit(X_train, y_train)

    model = CalibratedClassifierCV(base, method="sigmoid", cv=3)
    model.fit(X_train, y_train)

    proba = model.predict_proba(X_test)[:, 1]
    pred = (proba >= 0.5).astype(int)

    roc = roc_auc_score(y_test, proba)
    pr = average_precision_score(y_test, proba)

    print("max_time     :", max_time)
    print("snap_train   :", snap_train)
    print("snap_test    :", snap_test)
    print("train rows   :", len(train_ds), "churn:", float(y_train.mean()))
    print("test  rows   :", len(test_ds),  "churn:", float(y_test.mean()))
    print("ROC_AUC(test):", float(roc))
    print("PR_AUC (test):", float(pr))
    print("\nConfusion matrix (test @0.5):")
    print(confusion_matrix(y_test, pred))
    print("\nClassification report (test @0.5):")
    print(classification_report(y_test, pred, digits=4))

    art = PROJECT_ROOT / "artifacts"
    art.mkdir(exist_ok=True)

    joblib.dump(model, art / "churn_model_timesplit.joblib")
    (art / "feature_list_timesplit.json").write_text(json.dumps(list(X_train.columns), indent=2), encoding="utf-8")

    meta = {
        "max_time": str(max_time),
        "snapshot_train": str(snap_train),
        "snapshot_test": str(snap_test),
        "train_rows": int(len(train_ds)),
        "test_rows": int(len(test_ds)),
        "train_churn_rate": float(y_train.mean()),
        "test_churn_rate": float(y_test.mean()),
        "n_features": int(X_train.shape[1]),
        "roc_auc_test": float(roc),
        "pr_auc_test": float(pr),
    }
    (art / "train_meta_timesplit.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print("\nSaved: artifacts/churn_model_timesplit.joblib + feature_list_timesplit.json + train_meta_timesplit.json")


if __name__ == "__main__":
    main()
