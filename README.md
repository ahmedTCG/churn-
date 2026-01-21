# Churn Prediction Project

## Overview
This project builds a **customer churn prediction model** using historical interaction data.
The goal is to identify customers who are likely to become **inactive** based on their **past behavior**, enabling early intervention.

The pipeline:
- Processes raw interaction events
- Builds customer-level behavioral features at fixed time snapshots
- Labels churn based on future inactivity
- Trains and evaluates a time-aware churn model
- Scores customers and assigns risk buckets

---

## Churn Definition
A customer is labeled as **churned (`churn = 1`)** if they have **no interaction events** in the **30 days following a snapshot date**.

This is a **behavioral inactivity definition**, not an unsubscribe-based definition.

---

## Project Structure

.
├── data/
│   ├── raw/            # Raw input data
│   ├── processed/      # Cleaned data and model datasets
│   └── features/       # Customer-level feature snapshots
│
├── src/
│   ├── pipelines/      # End-to-end pipeline steps
│   ├── features/       # Feature engineering logic
│   ├── inference/      # Scoring and postprocessing
│   └── utils/          # Shared utilities
│
├── artifacts/          # Trained models and metadata
├── outputs/            # Scores and risk bucket outputs
├── notebooks/          # Exploratory and analysis notebooks
└── README.md

---

## Pipeline Steps

### 1. Data Processing
**Script:** `src/pipelines/01_make_processed.py`

- Loads raw interaction data
- Cleans timestamps and customer identifiers
- Standardizes interaction types
- Writes a clean event-level table


---

### 2. Feature Engineering
**Script:** `src/pipelines/02_make_features_snapshot_30d.py`

- Defines a **snapshot time**
- Uses only events **before the snapshot**
- Aggregates customer behavior into features:
  - Event counts (30/60/90 days)
  - Recency and activity windows
  - Frequency trends
  - Channel and engagement signals

Only **strong-signal interaction events** are used for feature creation.

Output:

---

### 3. Label Creation
**Script:** `src/pipelines/03_make_dataset_label_30d.py`

- Assigns churn labels based on **future inactivity**
- Ensures strict separation between:
  - Past behavior (features)
  - Future behavior (labels)

Output:


---

### 4. Model Training & Evaluation
**Script:** `src/pipelines/04_train_logreg_timesplit.py`

- Uses **time-based train/test split**
- Trains a **regularized logistic regression model**
- Evaluates using:
  - ROC-AUC
  - PR-AUC
  - Confusion matrix
  - Classification report

Artifacts saved:

artifacts/
├── churn_model_timesplit.joblib
├── feature_list_timesplit.json
└── train_meta_timesplit.json


---

### 5. Scoring & Postprocessing
**Scripts:**
- `src/inference/score_customers.py`
- `src/inference/postprocess_scores.py`

- Scores customers with churn probabilities
- Assigns risk buckets:
  - low
  - medium
  - high
  - critical

Outputs:

outputs/
├── churn_scores_timesplit.csv
├── churn_scores_timesplit_with_buckets.csv
├── churn_scores_timesplit_top5000.csv
├── churn_scores_timesplit_high.csv
└── churn_scores_timesplit_critical.csv

---

## Strong-Signal Events
Only selected interaction types are used to build features.
These represent **meaningful engagement signals** (email opens, clicks, sessions, orders, etc.).

The list is defined in:
src/features/strong_events.py

Customers with **no strong-signal events** are excluded from modeling.

---

## Running the Project

### Run the full pipeline
```bash
python -m src.pipelines.00_run_all

