from pathlib import Path
import pandas as pd
import numpy as np

from src.features.build_features import filter_strong_events

PROJECT_ROOT = Path(__file__).resolve().parents[2]
CHURN_WINDOW_DAYS = 30

def main():
    interactions_path = PROJECT_ROOT / "data" / "processed" / "interactions.parquet"
    df = pd.read_parquet(interactions_path)

    max_time = pd.to_datetime(df["event_time"]).max()
    snapshot_time = max_time - pd.Timedelta(days=CHURN_WINDOW_DAYS)
    future_end = snapshot_time + pd.Timedelta(days=CHURN_WINDOW_DAYS)

    df_hist = df[df["event_time"] <= snapshot_time].copy()
    customers = df_hist[["external_customerkey"]].drop_duplicates().reset_index(drop=True)

    df_strong = filter_strong_events(df)
    active_in_future = (
        df_strong[
            (df_strong["event_time"] > snapshot_time) &
            (df_strong["event_time"] <= future_end)
        ]["external_customerkey"]
        .drop_duplicates()
    )

    labels = customers.copy()
    labels["active_next_30d"] = labels["external_customerkey"].isin(active_in_future).astype(int)
    labels["churn_label"] = (1 - labels["active_next_30d"]).astype(int)

    features_path = PROJECT_ROOT / "data" / "features" / "customer_features_snapshot_30d.parquet"
    features = pd.read_parquet(features_path)

    dataset = features.merge(
        labels[["external_customerkey", "churn_label"]],
        on="external_customerkey",
        how="left"
    )

    dataset = dataset.dropna(subset=["churn_label"]).copy()
    dataset["churn_label"] = dataset["churn_label"].astype(int)

    dataset = dataset.replace([np.inf, -np.inf], np.nan)
    num_cols = dataset.select_dtypes(include=[np.number]).columns
    dataset[num_cols] = dataset[num_cols].fillna(0)

    out_path = PROJECT_ROOT / "data" / "processed" / "model_dataset_label_30d.parquet"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    dataset.to_parquet(out_path, index=False)

    print("snapshot_time:", snapshot_time)
    print("label mean:", dataset["churn_label"].mean())
    print("Wrote:", out_path, "rows:", len(dataset), "cols:", dataset.shape[1])

if __name__ == "__main__":
    main()
