# src/inference/score_customers.py
import json
import joblib
import pandas as pd
import numpy as np
from pathlib import Path

import warnings

# Silence known numeric-backend RuntimeWarnings during sklearn internal matmul.
warnings.filterwarnings("ignore", category=RuntimeWarning, message=".*matmul.*")
warnings.filterwarnings("ignore", category=RuntimeWarning, message=".*raw_prediction.*")
warnings.filterwarnings("ignore", category=RuntimeWarning, message=".*grad.*matmul.*")

from src.features.build_features import build_customer_features

PROJECT_ROOT = Path(__file__).resolve().parents[2]

def build_features_for_snapshot(df: pd.DataFrame, snapshot_time: pd.Timestamp) -> pd.DataFrame:
    return build_customer_features(df, snapshot_time=snapshot_time)


def main():
    data_path = PROJECT_ROOT / "data" / "processed" / "interactions.parquet"
    df = pd.read_parquet(data_path)

    # snapshot: use latest available time in data
    snapshot_time = pd.to_datetime(df["event_time"]).max()

    feats = build_features_for_snapshot(df, snapshot_time=snapshot_time)

    # Load model + feature list
    artifacts = PROJECT_ROOT / "artifacts"
    model = joblib.load(artifacts / "churn_model_timesplit.joblib")
    feature_list = json.loads((artifacts / "feature_list_timesplit.json").read_text(encoding="utf-8"))

    X = feats.set_index("external_customerkey")

    # align columns exactly like training
    for col in feature_list:
        if col not in X.columns:
            X[col] = 0
    X = X[feature_list]

    proba = model.predict_proba(X)[:, 1]
    assert np.isfinite(proba).all(), 'Non-finite probabilities produced'

    out = pd.DataFrame({
        "external_customerkey": X.index,
        "churn_probability": proba,
    }).sort_values("churn_probability", ascending=False)

    out_path = PROJECT_ROOT / "outputs" / "churn_scores_timesplit.csv"
    out_path.parent.mkdir(exist_ok=True)
    out.to_csv(out_path, index=False)

    print("Saved:", out_path, "rows:", len(out))

if __name__ == "__main__":
    main()
