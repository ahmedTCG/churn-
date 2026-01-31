import pandas as pd
from pathlib import Path

RAW_PATH = Path("data/raw/ubdated.csv")
OUT_PATH = Path("data/processed/interactions.parquet")


def make_processed_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """
    Production-friendly cleaning function.
    EXACT same logic used by the pipeline.
    No IO.
    """
    df = df.copy()

    # ===== CLEANING LOGIC (KEEP THIS IDENTICAL TO TRAINING) =====
    df["event_time"] = pd.to_datetime(df["event_time"], errors="coerce")
    df["external_customerkey"] = df["external_customerkey"].astype(str).str.strip()
    df["interaction_type"] = df["interaction_type"].astype(str).str.strip()

    if "amount" in df.columns:
        df["amount"] = pd.to_numeric(df["amount"], errors="coerce")

    df = df.dropna(
        subset=["external_customerkey", "event_time", "interaction_type"]
    )
    # ===========================================================

    return df


def main():
    raw_df = pd.read_csv(RAW_PATH)
    processed_df = make_processed_dataframe(raw_df)

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    processed_df.to_parquet(OUT_PATH, index=False)

    print(f"OK → {OUT_PATH}")


if __name__ == "__main__":
    main()
