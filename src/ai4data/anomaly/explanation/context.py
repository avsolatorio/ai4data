"""Context extraction for anomaly explanation prompts.

Extracts time-windowed context around anomalies for LLM consumption.
"""

import json
from typing import Any, Dict, List

import pandas as pd

# Canonical column names
INDICATOR_ID = "indicator_id"
INDICATOR_NAME = "indicator_name"
GEOGRAPHY_ID = "geography_id"
GEOGRAPHY_NAME = "geography_name"
PERIOD = "period"
VALUE = "value"
IS_IMPUTED = "is_imputed"
OUTLIER_COUNT = "outlier_count"


def extract_anomaly_contexts(
    df: pd.DataFrame,
    geography_name_map: Dict[str, str],
    indicator_name_map: Dict[str, str],
    period_window: int = 3,
    min_outlier_count: int = 3,
    period_col: str = PERIOD,
    value_col: str = VALUE,
    is_imputed_col: str = IS_IMPUTED,
    outlier_count_col: str = OUTLIER_COUNT,
) -> List[Dict[str, Any]]:
    """Extract time-windowed context around anomalies for each (indicator, geography) series.

    For each anomalous period, builds a window of [period - window, period + window]
    and merges overlapping windows into contiguous ranges. Each range yields one
    context dict with Indicator, Geography, and Series (list of {period, value, is_imputed}).

    Parameters
    ----------
    df : pd.DataFrame
        Long-format DataFrame for a single (indicator_id, geography_id) with
        period, value, is_imputed, and outlier_count columns.
    geography_name_map : dict
        Maps geography_id -> geography_name.
    indicator_name_map : dict
        Maps indicator_id -> indicator_name.
    period_window : int
        Number of periods to include before/after each anomaly.
    min_outlier_count : int
        Minimum outlier_count for a period to be considered an anomaly.
    period_col, value_col, is_imputed_col, outlier_count_col : str
        Column names (support both canonical and legacy names).

    Returns
    -------
    list of dict
        Each dict has keys: "Indicator", "Country", "Series".
        "Series" is a list of {"YEAR": int, "VALUE": float, "Imputed": bool}.
    """
    if df[INDICATOR_ID].nunique() != 1 or df[GEOGRAPHY_ID].nunique() != 1:
        raise ValueError(
            f"DataFrame must have exactly one {INDICATOR_ID} and one {GEOGRAPHY_ID}"
        )

    geography_id = df[GEOGRAPHY_ID].iloc[0]
    indicator_id = df[INDICATOR_ID].iloc[0]
    geography_name = geography_name_map.get(geography_id, str(geography_id))
    indicator_name = indicator_name_map.get(indicator_id, str(indicator_id))

    df = df.sort_values(period_col)
    min_period = int(df[period_col].min())
    max_period = int(df[period_col].max())

    # Periods flagged as anomalies
    anomaly_mask = (df[outlier_count_col] >= min_outlier_count) & (~df[is_imputed_col])
    anomaly_periods = df.loc[anomaly_mask, period_col].dropna().astype(int)

    if anomaly_periods.empty:
        return []

    low_periods = anomaly_periods - period_window
    high_periods = anomaly_periods + period_window
    ranges = [
        list(range(max(min_period, int(lo)), min(int(hi), max_period) + 1))
        for lo, hi in zip(low_periods, high_periods)
    ]

    # Merge overlapping ranges
    valid_periods: List[List[int]] = []
    current = set(ranges[0]) if ranges else set()
    for r in ranges[1:]:
        rset = set(r)
        if current.intersection(rset):
            current = current.union(rset)
        else:
            valid_periods.append(sorted(current))
            current = rset
    if current:
        valid_periods.append(sorted(current))

    contexts = []
    for periods in valid_periods:
        sub = df[df[period_col].isin(periods)][[period_col, value_col, is_imputed_col]]
        sub = sub.rename(columns={period_col: "YEAR", value_col: "VALUE", is_imputed_col: "Imputed"})
        series_data = sub.to_dict(orient="records")
        # Ensure correct types for JSON
        for row in series_data:
            row["YEAR"] = int(row["YEAR"]) if pd.notna(row["YEAR"]) else None
            row["VALUE"] = float(row["VALUE"]) if pd.notna(row["VALUE"]) else None
            row["Imputed"] = bool(row["Imputed"])
        context = {
            "Indicator": indicator_name,
            "Country": geography_name,
            "Series": series_data,
        }
        contexts.append(context)

    return contexts
