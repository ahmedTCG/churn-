from pathlib import Path
import numpy as np
import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, average_precision_score, precision_recall_fscore_support

PROJECT_ROOT = Path(__file__).resolve().parents[2]

TARGET = "churn_label"
ID_COL = "external_customerkey"
DROP_FEATURES = ["total_revenue", "avg_order_value"]

def main():
    dataset_path = PROJECT_ROOT / "data" / "processed" / "model_dataset_label_30d.parquet"
    model_path = PROJECT_ROOT / "artifacts" / "churn_model_v1.joblib"
    feature_list_path = PROJECT_ROOT / "artifacts" / "feature_list.json"

    df = pd.read_parquet(dataset_path)
    y = df[TARGET].astype(int)

    X = df.drop(columns=[c for c in [ID_COL, TARGET] if c in df.columns]).copy()
    X = X.drop(columns=DROP_FEATURES, errors="ignore")
    X = X.replace([np.inf, -np.inf], np.nan).fillna(0)
    X = X.select_dtypes(include=[np.number])

    # Same split style as training
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # Load model + feature list used during training
    model = joblib.load(model_path)
    feature_list = pd.read_json(feature_list_path, typ='series').tolist()

    # Align feature columns exactly like training
    X_test_aligned = X_test.copy()
    for col in feature_list:
        if col not in X_test_aligned.columns:
            X_test_aligned[col] = 0
    X_test_aligned = X_test_aligned[feature_list]

    proba = model.predict_proba(X_test_aligned)[:, 1]

    roc = roc_auc_score(y_test, proba)
    pr = average_precision_score(y_test, proba)

    rows = []
    for thr in [round(x, 2) for x in np.linspace(0.1, 0.9, 9)]:
        pred = (proba >= thr).astype(int)
        prec, rec, f1, _ = precision_recall_fscore_support(
            y_test, pred, average="binary", zero_division=0
        )
        pos_rate = float(pred.mean())
        rows.append({
            "threshold": thr,
            "precision": float(prec),
            "recall": float(rec),
            "f1": float(f1),
            "predicted_positive_rate": pos_rate,
        })

    report = pd.DataFrame(rows)
    out = PROJECT_ROOT / "artifacts" / "eval_report_v1.csv"
    report.to_csv(out, index=False)

    print("ROC_AUC:", roc)
    print("PR_AUC :", pr)
    print("Wrote :", out)
    print(report)

if __name__ == "__main__":
    main()
