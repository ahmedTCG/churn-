import pandas as pd
import numpy as np

# -----------------------------------
# Strong signal events (from EDA)
# -----------------------------------
from src.features.strong_events import STRONG_SIGNAL_EVENTS


def filter_strong_events(df: pd.DataFrame) -> pd.DataFrame:
    """
    Keep only strong-signal interaction events.
    """
    if "interaction_type" not in df.columns:
        raise ValueError("Expected column 'interaction_type' not found in DataFrame")

    return df[df["interaction_type"].isin(STRONG_SIGNAL_EVENTS)].copy()



import pandas as pd
from typing import Optional

def build_recency_feature(
    df: pd.DataFrame,
    snapshot_time: Optional[pd.Timestamp] = None
) -> pd.DataFrame:
    """
    Build customer-level recency feature:
    number of days since last interaction.
    """

    # If no snapshot time is provided, use the latest event_time in data
    if snapshot_time is None:
        snapshot_time = df["event_time"].max()

    # Get last interaction per customer
    recency_df = (
        df.groupby("external_customerkey")["event_time"]
          .max()
          .reset_index()
    )

    # Compute recency in days
    recency_df["recency_days"] = (
        snapshot_time - recency_df["event_time"]
    ).dt.days

    return recency_df[["external_customerkey", "recency_days"]]

import pandas as pd
from typing import Optional

def build_frequency_feature(
    df: pd.DataFrame,
    snapshot_time: Optional[pd.Timestamp] = None,
    window_days: int = 30
) -> pd.DataFrame:
    """
    Customer-level frequency feature:
    number of interactions in the last `window_days` before snapshot_time.
    """
    if snapshot_time is None:
        snapshot_time = df["event_time"].max()

    window_start = snapshot_time - pd.Timedelta(days=window_days)

    df_win = df[(df["event_time"] > window_start) & (df["event_time"] <= snapshot_time)].copy()

    freq = (
        df_win.groupby("external_customerkey")
        .size()
        .rename(f"n_events_last_{window_days}d")
        .reset_index()
    )

    return freq


import pandas as pd
from typing import Optional

def build_active_days_feature(
    df: pd.DataFrame,
    snapshot_time: Optional[pd.Timestamp] = None,
    window_days: int = 30
) -> pd.DataFrame:
    """
    Customer-level feature:
    number of distinct active days in the last `window_days`.
    """
    if snapshot_time is None:
        snapshot_time = df["event_time"].max()

    window_start = snapshot_time - pd.Timedelta(days=window_days)

    df_window = df[(df["event_time"] > window_start) & (df["event_time"] <= snapshot_time)].copy()
    df_window["event_date"] = df_window["event_time"].dt.date

    out = (
        df_window.groupby("external_customerkey")["event_date"]
        .nunique()
        .rename(f"active_days_last_{window_days}d")
        .reset_index()
    )

    return out

import pandas as pd

def build_order_value_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Customer-level order/value features (orders only):
    - n_orders
    - total_revenue
    - avg_order_value
    """
    df_orders = df[df["interaction_type"] == "order"].copy()
    df_orders["amount"] = pd.to_numeric(df_orders["amount"], errors="coerce")

    agg = (
        df_orders.groupby("external_customerkey")["amount"]
        .agg(
            n_orders="count",
            total_revenue="sum",
            avg_order_value="mean",
        )
        .reset_index()
    )

    # customers with no orders won't appear here; fill later when merging features
    return agg

import pandas as pd
from typing import Optional, List

def build_event_type_counts_feature(
    df: pd.DataFrame,
    event_types: List[str],
    snapshot_time: Optional[pd.Timestamp] = None,
    window_days: int = 30
) -> pd.DataFrame:
    """
    Customer-level feature:
    counts of each event type in the last `window_days` before snapshot_time.
    Returns wide table: one row per customer + one column per event type.
    """
    if snapshot_time is None:
        snapshot_time = df["event_time"].max()

    window_start = snapshot_time - pd.Timedelta(days=window_days)
    df_window = df[(df["event_time"] > window_start) & (df["event_time"] <= snapshot_time)]

    df_window = df_window[df_window["interaction_type"].isin(event_types)]

    counts = (
        df_window.groupby(["external_customerkey", "interaction_type"])
        .size()
        .unstack(fill_value=0)
        .reset_index()
    )

    # nice column names
    rename_map = {e: f"cnt_{e}_last_{window_days}d" for e in event_types if e in counts.columns}
    counts = counts.rename(columns=rename_map)

    return counts

import pandas as pd

def build_frequency_trend_features(df_features: pd.DataFrame) -> pd.DataFrame:
    """
    Frequency trend features based on existing frequency windows.
    Assumes the following columns already exist:
    - n_events_last_30d
    - n_events_last_60d
    - n_events_last_90d
    """

    out = df_features[["external_customerkey"]].copy()

    out["freq_trend_30_60"] = (
        df_features["n_events_last_30d"] /
        (df_features["n_events_last_60d"] + 1)
    )

    out["freq_trend_60_90"] = (
        df_features["n_events_last_60d"] /
        (df_features["n_events_last_90d"] + 1)
    )

    return out

import pandas as pd

def build_activity_ratio_features(df_features: pd.DataFrame) -> pd.DataFrame:
    """
    Activity ratio features based on active days.
    Assumes the following column exists:
    - active_days_last_30d
    """

    out = df_features[["external_customerkey"]].copy()

    out["active_ratio_30d"] = (
        df_features["active_days_last_30d"] / 30
    )

    return out

import pandas as pd
from typing import List

def build_inactivity_flag_features(
    df_features: pd.DataFrame,
    thresholds: List[int] = [14, 30]
) -> pd.DataFrame:
    """
    Binary inactivity flags based on recency_days.
    Example:
    - inactive_14d = recency_days > 14
    - inactive_30d = recency_days > 30
    """

    out = df_features[["external_customerkey"]].copy()

    for t in thresholds:
        out[f"inactive_{t}d"] = (df_features["recency_days"] > t).astype(int)

    return out


# ============================================================
# Unified customer feature builder (training + inference)
# ============================================================

def build_customer_features(
    df: pd.DataFrame,
    snapshot_time: pd.Timestamp,
    churn_window_days: int = 30,
) -> pd.DataFrame:
    """
    Build full customer-level feature table at a given snapshot_time.
    Used by both training and inference to prevent feature drift.
    """

    df = df.copy()
    df["event_time"] = pd.to_datetime(df["event_time"], errors="coerce")
    df = df.dropna(subset=["external_customerkey", "event_time", "interaction_type"])

    df_hist = df[df["event_time"] <= snapshot_time].copy()
    customers = df_hist[["external_customerkey"]].drop_duplicates()

    df_strong_hist = filter_strong_events(df_hist)

    # Base features
    recency = build_recency_feature(df_strong_hist, snapshot_time=snapshot_time)

    freq_30 = build_frequency_feature(df_strong_hist, snapshot_time, 30)
    freq_60 = build_frequency_feature(df_strong_hist, snapshot_time, 60)
    freq_90 = build_frequency_feature(df_strong_hist, snapshot_time, 90)

    active_30 = build_active_days_feature(df_strong_hist, snapshot_time, 30)
    active_60 = build_active_days_feature(df_strong_hist, snapshot_time, 60)
    active_90 = build_active_days_feature(df_strong_hist, snapshot_time, 90)

    mix_30 = build_event_type_counts_feature(df_strong_hist, STRONG_SIGNAL_EVENTS, snapshot_time, 30)
    mix_60 = build_event_type_counts_feature(df_strong_hist, STRONG_SIGNAL_EVENTS, snapshot_time, 60)
    mix_90 = build_event_type_counts_feature(df_strong_hist, STRONG_SIGNAL_EVENTS, snapshot_time, 90)

    order_feats = build_order_value_features(df_strong_hist)

    features = (
        customers
        .merge(recency, on="external_customerkey", how="left")
        .merge(freq_30, on="external_customerkey", how="left")
        .merge(freq_60, on="external_customerkey", how="left")
        .merge(freq_90, on="external_customerkey", how="left")
        .merge(active_30, on="external_customerkey", how="left")
        .merge(active_60, on="external_customerkey", how="left")
        .merge(active_90, on="external_customerkey", how="left")
        .merge(mix_30, on="external_customerkey", how="left")
        .merge(mix_60, on="external_customerkey", how="left")
        .merge(mix_90, on="external_customerkey", how="left")
        .merge(order_feats, on="external_customerkey", how="left")
    )

    # Post-processing
    freq_trends = build_frequency_trend_features(features)
    activity_ratio = build_activity_ratio_features(features)
    inactive_flags = build_inactivity_flag_features(features, thresholds=[14, 30])

    features = (
        features
        .merge(freq_trends, on="external_customerkey", how="left")
        .merge(activity_ratio, on="external_customerkey", how="left")
        .merge(inactive_flags, on="external_customerkey", how="left")
    )

    # Fill NAs
    count_cols = [c for c in features.columns if c.startswith(("n_events_last_", "active_days_last_", "cnt_"))]
    features[count_cols] = features[count_cols].fillna(0)

    for c in ["freq_trend_30_60", "freq_trend_60_90", "active_ratio_30d"]:
        if c in features.columns:
            features[c] = features[c].fillna(0)

    for c in ["inactive_14d", "inactive_30d", "n_orders", "total_revenue", "avg_order_value"]:
        if c in features.columns:
            features[c] = features[c].fillna(0)

    features["has_any_strong_event_hist"] = (~features["recency_days"].isna()).astype(int)
    features["recency_days"] = features["recency_days"].fillna(999)

    features = features.replace([np.inf, -np.inf], np.nan).fillna(0)

    return features
