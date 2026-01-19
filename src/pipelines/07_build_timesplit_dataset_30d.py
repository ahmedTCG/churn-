from pathlib import Path
import pandas as pd
import numpy as np

from src.features.build_features import build_customer_features, filter_strong_events

PROJECT_ROOT = Path(__file__).resolve().parents[2]
CHURN_WINDOW_DAYS = 30

def main():
    data_path = PROJECT_ROOT / "data" / "processed" / "interactions.parquet"
    df = pd.read_parquet(data_path)
    df["event_time"] = pd.to_datetime(df["event_time"], errors="coerce")
    df = df.dropna(subset=["external_customerkey", "event_time", "interaction_type"]).copy()

    min_date = df["event_time"].min()
    max_date = df["event_time"].max()
    latest_snapshot = max_date - pd.Timedelta(days=CHURN_WINDOW_DAYS)

    # Month-end snapshots (use 'M' style but explicit month-end to avoid confusion)
    snapshots = pd.date_range(
        start=min_date.normalize() + pd.offsets.MonthEnd(0),
        end=latest_snapshot.normalize(),
        freq="M",
    )

    df_strong = filter_strong_events(df)

    all_rows = []
    for snap in snapshots:
        future_end = snap + pd.Timedelta(days=CHURN_WINDOW_DAYS)

        # Customers seen up to snapshot
        customers = df.loc[df["event_time"] <= snap, ["external_customerkey"]].drop_duplicates()

        # Active in future (strong events only)
        active_in_future = (
            df_strong.loc[
                (df_strong["event_time"] > snap) & (df_strong["event_time"] <= future_end),
                "external_customerkey"
            ]
            .drop_duplicates()
        )

        labels = customers.copy()
        labels["snapshot_time"] = snap
        labels["active_next_30d"] = labels["external_customerkey"].isin(active_in_future).astype(int)
        labels["churn_label"] = (1 - labels["active_next_30d"]).astype(int)

        feats = build_customer_features(df, snapshot_time=snap)
        feats["snapshot_time"] = snap

        dataset = feats.merge(
            labels[["external_customerkey", "snapshot_time", "churn_label"]],
            on=["external_customerkey", "snapshot_time"],
            how="inner",
        )

        dataset = dataset.replace([np.inf, -np.inf], np.nan)
        num_cols = dataset.select_dtypes(include=[np.number]).columns
        dataset[num_cols] = dataset[num_cols].fillna(0)

        all_rows.append(dataset)

        print("snapshot:", snap, "rows:", len(dataset), "label_mean:", float(dataset["churn_label"].mean()))

    out = pd.concat(all_rows, ignore_index=True)
    out_path = PROJECT_ROOT / "data" / "processed" / "model_dataset_timesplit_30d.parquet"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out.to_parquet(out_path, index=False)

    print("Wrote:", out_path)
    print("Total rows:", len(out), "cols:", out.shape[1])
    print("Overall label mean:", float(out["churn_label"].mean()))

if __name__ == "__main__":
    main()
