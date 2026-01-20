import subprocess
import sys

STEPS = [
    ["python", "-m", "src.pipelines.01_make_processed"],
    ["python", "-m", "src.pipelines.02_make_features_snapshot_30d"],
    ["python", "-m", "src.pipelines.03_make_dataset_label_30d"],
    ["python", "-m", "src.pipelines.04_train_logreg_timesplit"],
    ["python", "-m", "src.inference.score_customers"],
    ["python", "-m", "src.inference.postprocess_scores"],
]

def run(cmd):
    print("\n" + "=" * 80)
    print("RUN:", " ".join(cmd))
    print("=" * 80)
    subprocess.run(cmd, check=True)

def main():
    for cmd in STEPS:
        run(cmd)
    print("\nPIPELINE COMPLETED ✅")

if __name__ == "__main__":
    main()
