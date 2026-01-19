from pathlib import Path
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]

RAW_PATH = PROJECT_ROOT / "data" / "raw" / "customer_interactions_fact_2_years.csv"
OUT_PATH = PROJECT_ROOT / "data" / "processed" / "interactions.parquet"
OUT_PATH.parent.mkdir(parents=True, exist_ok=True)

def main():
    if not RAW_PATH.exists():
        raise FileNotFoundError(f"Missing raw file: {RAW_PATH}")

    df = pd.read_csv(RAW_PATH, parse_dates=["event_time"], low_memory=False)

    # Ensure datetime
    df["event_time"] = pd.to_datetime(df["event_time"], errors="coerce")

    # Drop critical nulls safely BEFORE casting
    df = df.dropna(subset=["external_customerkey", "event_time", "interaction_type"]).copy()

    # Standardize key string cols
    df["interaction_type"] = df["interaction_type"].astype("string").str.strip()
    df["external_customerkey"] = df["external_customerkey"].astype("string").str.strip()

    for c in ["channel", "shop", "incoming_outgoing"]:
        if c in df.columns:
            df[c] = df[c].astype("string").str.strip()

    df.to_parquet(OUT_PATH, index=False)
    print("Wrote:", OUT_PATH, "rows:", len(df))

if __name__ == "__main__":
    main()
