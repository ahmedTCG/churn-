import argparse
import importlib
from pathlib import Path

import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.calibration import CalibratedClassifierCV
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import roc_auc_score, average_precision_score


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


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--step-days", type=int, default=30, help="Snapshot step (days). Default=30.")
    p.add_argument("--lookback-days", type=int, default=90, help="Max feature lookback required (days). Default=90.")
    p.add_argument("--max-evals", type=int, default=0, help="Limit number of evaluations (0 = all possible).")
    p.add_argument("--out", type=str, default="outputs/rolling_snapshot_eval.csv", help="Output CSV path.")
    p.add_argument("--min-roc", type=float, default=None, help="If set, fail (exit 1) if worst ROC < min-roc.")
    p.add_argument("--min-pr", type=float, default=None, help="If set, fail (exit 1) if worst PR < min-pr.")
    args = p.parse_args()

    train_mod = importlib.import_module("src.pipelines.04_train_logreg_timesplit")
    TARGET = train_mod.TARGET
    W = args.step_days

    # Load event data (same source your pipeline uses)
    df = pd.read_parquet(train_mod.PROJECT_ROOT / "data" / "processed" / "interactions.parquet")
    df["event_time"] = pd.to_datetime(df["event_time"], errors="coerce")
    df = df.dropna(subset=["external_customerkey", "event_time", "interaction_type"]).copy()

    min_time = df["event_time"].min()
    max_time = df["event_time"].max()

    # Latest test snapshot must allow a full future label window (W days)
    latest_test_snap = max_time - pd.Timedelta(days=W)

    # Earliest train snapshot must allow full lookback history for features
    earliest_train_snap_allowed = min_time + pd.Timedelta(days=args.lookback_days)

    def make_X(ds_features_only):
        if hasattr(train_mod, "make_X"):
            return train_mod.make_X(ds_features_only)
        if hasattr(train_mod, "transform_features"):
            return train_mod.transform_features(ds_features_only)
        raise AttributeError("Can't find make_X or transform_features in src.pipelines.04_train_logreg_timesplit")

    rows = []
    test_snap = latest_test_snap
    n = 0

    while True:
        train_snap = test_snap - pd.Timedelta(days=W)

        if train_snap < earliest_train_snap_allowed:
            break

        train_ds = train_mod.make_dataset(df, train_snap)
        test_ds = train_mod.make_dataset(df, test_snap)

        X_train = make_X(train_ds.drop(columns=[TARGET]))
        y_train = train_ds[TARGET].astype(int).values

        X_test = make_X(test_ds.drop(columns=[TARGET]))
        y_test = test_ds[TARGET].astype(int).values

        # Align columns (safety)
        X_test = X_test.reindex(columns=X_train.columns, fill_value=0)

        model = fit_model(X_train, y_train)
        proba = model.predict_proba(X_test)[:, 1]

        rows.append({
            "train_snapshot": str(train_snap),
            "test_snapshot": str(test_snap),
            "train_rows": int(len(train_ds)),
            "test_rows": int(len(test_ds)),
            "train_churn_rate": float(y_train.mean()),
            "test_churn_rate": float(y_test.mean()),
            "roc_auc": float(roc_auc_score(y_test, proba)),
            "pr_auc": float(average_precision_score(y_test, proba)),
        })

        test_snap = test_snap - pd.Timedelta(days=W)
        n += 1
        if args.max_evals and n >= args.max_evals:
            break

    out = pd.DataFrame(rows).sort_values("test_snapshot").reset_index(drop=True)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(out_path, index=False)

    print("Event time range:", min_time, "→", max_time)
    print("Evaluations run:", len(out))
    print(out.to_string(index=False))
    print("\nSaved:", out_path)

    # Optional CI gates
    if args.min_roc is not None:
        worst_roc = out["roc_auc"].min()
        print(f"\nWorst ROC: {worst_roc:.6f} (min required: {args.min_roc})")
        if worst_roc < args.min_roc:
            raise SystemExit(1)

    if args.min_pr is not None:
        worst_pr = out["pr_auc"].min()
        print(f"Worst PR:  {worst_pr:.6f} (min required: {args.min_pr})")
        if worst_pr < args.min_pr:
            raise SystemExit(1)


if __name__ == "__main__":
    main()
