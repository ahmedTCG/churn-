from pathlib import Path
import pandas as pd
import numpy as np

from src.features.build_features import build_customer_features

PROJECT_ROOT = Path(__file__).resolve().parents[2]

CHURN_WINDOW_DAYS = 30

def main():
    data_path = PROJECT_ROOT / "data" / "processed" / "interactions.parquet"
    df = pd.read_parquet(data_path)

    max_time = pd.to_datetime(df["event_time"]).max()
    snapshot_time = max_time - pd.Timedelta(days=CHURN_WINDOW_DAYS)

    # Build features at snapshot_time using the unified builder
    features = build_customer_features(df, snapshot_time=snapshot_time)

    out_path = PROJECT_ROOT / "data" / "features" / f"customer_features_snapshot_{CHURN_WINDOW_DAYS}d.parquet"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    features.to_parquet(out_path, index=False)

    print("snapshot_time:", snapshot_time)
    print("Wrote:", out_path, "rows:", len(features), "cols:", features.shape[1])

if __name__ == "__main__":
    main()
