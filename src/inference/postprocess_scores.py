from pathlib import Path
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]

def main():
    in_path = PROJECT_ROOT / "outputs" / "churn_scores_v1.csv"
    if not in_path.exists():
        raise FileNotFoundError(f"Missing scores file: {in_path}")

    scores = pd.read_csv(in_path)

    scores["risk_bucket"] = pd.cut(
        scores["churn_probability"],
        bins=[0, 0.5, 0.7, 0.85, 1.0],
        labels=["low", "medium", "high", "critical"],
        include_lowest=True,
    )

    out_dir = PROJECT_ROOT / "outputs"
    out_dir.mkdir(exist_ok=True)

    # With buckets
    with_buckets = out_dir / "churn_scores_v1_with_buckets.csv"
    scores.to_csv(with_buckets, index=False)

    # Sorted
    scores_sorted = scores.sort_values("churn_probability", ascending=False)

    # Top 5000
    top5000 = out_dir / "churn_scores_v1_top5000.csv"
    scores_sorted.head(5000).to_csv(top5000, index=False)

    # Split by bucket
    critical = out_dir / "churn_scores_v1_critical.csv"
    high = out_dir / "churn_scores_v1_high.csv"
    scores_sorted[scores_sorted["risk_bucket"] == "critical"].to_csv(critical, index=False)
    scores_sorted[scores_sorted["risk_bucket"] == "high"].to_csv(high, index=False)

    print("Wrote:")
    print("-", with_buckets)
    print("-", top5000)
    print("-", critical)
    print("-", high)
    print("Bucket counts:")
    print(scores["risk_bucket"].value_counts(dropna=False))

if __name__ == "__main__":
    main()
