import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.calibration import CalibratedClassifierCV
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import roc_auc_score, average_precision_score

# Make imports work when running as a script
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import importlib
train_mod = importlib.import_module("src.pipelines.04_train_logreg_timesplit")

TARGET = train_mod.TARGET
W = train_mod.CHURN_WINDOW_DAYS

# Must match your feature max window
LOOKBACK_DAYS = 90

def fit_model(X_train, y_train, random_state: int = 42):
    base = Pipeline([
        ("scaler", StandardScaler(with_mean=False)),
        ("clf", SGDClassifier(
            loss="log_loss",
            penalty="elasticnet",
            alpha=1e-4,
            l1_ratio=0.15,
            max_iter=2000,
            tol=1e-3,
            random_state=random_state,
        )),
    ])
    model = CalibratedClassifierCV(base, method="sigmoid", cv=3)
    model.fit(X_train, y_train)
    return model

def make_X(ds_features_only):
    if hasattr(train_mod, "make_X"):
        return train_mod.make_X(ds_features_only)
    if hasattr(train_mod, "transform_features"):
        return train_mod.transform_features(ds_features_only)
    raise AttributeError("Can't find make_X or transform_features")

def pooled_train_ds(df, test_snap):
    # Use the pooled builder you added in the pipeline if it exists
    if hasattr(train_mod, "build_pooled_train_dataset"):
        return train_mod.build_pooled_train_dataset(df, test_snap)

    # Fallback: pool snapshots manually (same logic)
    min_time = df["event_time"].min()
    earliest_allowed = min_time + pd.Timedelta(days=LOOKBACK_DAYS)

    parts = []
    k = 1
    while True:
        snap = test_snap - pd.Timedelta(days=k * W)
        if snap < earliest_allowed:
            break
        ds = train_mod.make_dataset(df, snap).copy()
        ds["_snapshot_time"] = snap
        parts.append(ds)
        k += 1
    if not parts:
        raise ValueError("No snapshots available for pooled training")
    return pd.concat(parts, ignore_index=True)

def main():
    df = pd.read_parquet(train_mod.PROJECT_ROOT / "data" / "processed" / "interactions.parquet")
    df["event_time"] = pd.to_datetime(df["event_time"], errors="coerce")
    df = df.dropna(subset=["external_customerkey", "event_time", "interaction_type"]).copy()

    min_time = df["event_time"].min()
    max_time = df["event_time"].max()

    latest_test_snap = max_time - pd.Timedelta(days=W)
    earliest_train_allowed = min_time + pd.Timedelta(days=LOOKBACK_DAYS)

    rows = []
    test_snap = latest_test_snap
    while True:
        # we still report the "paired" train snapshot for readability
        train_snap = test_snap - pd.Timedelta(days=W)
        if train_snap < earliest_train_allowed:
            break

        train_ds = pooled_train_ds(df, test_snap)
        test_ds  = train_mod.make_dataset(df, test_snap)

        X_train = make_X(train_ds.drop(columns=[TARGET]))
        y_train = train_ds[TARGET].astype(int).values

        X_test = make_X(test_ds.drop(columns=[TARGET]))
        y_test = test_ds[TARGET].astype(int).values

        X_test = X_test.reindex(columns=X_train.columns, fill_value=0)

        model = fit_model(X_train, y_train)
        proba = model.predict_proba(X_test)[:, 1]

        rows.append({
            "train_snapshot_label": str(train_snap),
            "test_snapshot": str(test_snap),
            "train_rows": int(len(train_ds)),
            "test_rows": int(len(test_ds)),
            "train_churn_rate": float(y_train.mean()),
            "test_churn_rate": float(y_test.mean()),
            "roc_auc": float(roc_auc_score(y_test, proba)),
            "pr_auc": float(average_precision_score(y_test, proba)),
        })

        test_snap = test_snap - pd.Timedelta(days=W)

    out = pd.DataFrame(rows).sort_values("test_snapshot").reset_index(drop=True)

    print("Event time range:", min_time, "→", max_time)
    print("Evaluations run:", len(out))
    print(out.to_string(index=False))

    out_path = train_mod.PROJECT_ROOT / "outputs" / "rolling_snapshot_eval_pooled.csv"
    out.to_csv(out_path, index=False)
    print("\nSaved:", out_path)

if __name__ == "__main__":
    main()
