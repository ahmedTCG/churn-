#!/usr/bin/env python3

import argparse
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]

RAW_PATH = ROOT / "data/raw/customer_interactions_fact_2_years.csv"
SCORES_OUT = ROOT / "outputs/churn_scores_timesplit.csv"


def run(cmd):
    print("RUN:", " ".join(cmd))
    subprocess.run(cmd, check=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--csv",
        required=True,
        help="Raw CSV (same schema as training data)"
    )
    parser.add_argument(
        "--out",
        default=str(SCORES_OUT),
        help="Output predictions CSV"
    )
    args = parser.parse_args()

    input_csv = Path(args.csv)
    if not input_csv.exists():
        raise FileNotFoundError(input_csv)

    # 1) place raw data where pipeline expects it
    RAW_PATH.parent.mkdir(parents=True, exist_ok=True)
    RAW_PATH.write_bytes(input_csv.read_bytes())

    # 2) cleaning
    run([sys.executable, "-m", "src.pipelines.01_make_processed"])

    # 3) features
    run([sys.executable, "-m", "src.pipelines.02_make_features_snapshot_30d"])

    # 4) inference only (uses trained model)
    run([sys.executable, "-m", "src.inference.score_customers"])

    # 5) copy output
    final_out = Path(args.out)
    final_out.parent.mkdir(parents=True, exist_ok=True)
    final_out.write_bytes(SCORES_OUT.read_bytes())

    print(f"OK → {final_out}")


if __name__ == "__main__":
    main()
