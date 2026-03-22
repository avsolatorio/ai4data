"""Reviewer-friendly output format for anomaly explanation pipeline.

Designed for easy integration into review applications: combines explanations
with timeseries data for display.
"""

from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd


# Schema for app-consumable output
REVIEW_OUTPUT_SCHEMA = {
    "version": "1.0",
    "description": "Anomaly explanations + timeseries for reviewer app",
    "items": {
        "indicator_code": "str",
        "indicator_name": "str",
        "geography_code": "str",
        "geography_name": "str",
        "window_start": "int",
        "window_end": "int",
        "window_str": "str",
        "timeseries": [{"period": "int", "value": "float", "is_imputed": "bool"}],
        "explanation": {
            "is_anomaly": "bool",
            "classification": "str",
            "confidence": "float",
            "explanation": "str",
            "evidence_strength": "str",
            "evidence_sources": "list",
            "source": "str",
        },
    },
}


def to_review_format(
    anomalies_df: pd.DataFrame,
    timeseries_df: Optional[pd.DataFrame] = None,
    indicator_col: str = "indicator_code",
    geography_col: str = "country_code",
    window_col: str = "window",
) -> List[Dict[str, Any]]:
    """Convert anomaly DataFrame to review-app-friendly format.

    Parameters
    ----------
    anomalies_df : pd.DataFrame
        Output from parse_batch_output or harmonize_explanations.
    timeseries_df : pd.DataFrame, optional
        Canonical long-format timeseries. If provided, each item includes
        a "timeseries" array for the anomaly window.
    indicator_col, geography_col, window_col : str
        Column names for indicator, geography, and window.

    Returns
    -------
    list of dict
        One dict per anomaly, suitable for JSON export and app consumption.
    """
    items = []
    for _, row in anomalies_df.iterrows():
        window = row.get(window_col, row.get("window", []))
        if isinstance(window, (list, tuple)) and len(window) >= 2:
            start, end = int(window[0]), int(window[1])
        else:
            start = end = None

        item = {
            "indicator_code": row.get(indicator_col, row.get("indicator_code", "")),
            "indicator_name": row.get("indicator", row.get("indicator_name", "")),
            "geography_code": row.get(geography_col, row.get("country_code", "")),
            "geography_name": row.get("country", row.get("geography_name", "")),
            "window_start": start,
            "window_end": end,
            "window_str": row.get("window_str", f"{start}-{end}" if start is not None else ""),
            "timeseries": [],
            "explanation": {
                "is_anomaly": row.get("is_anomaly", False),
                "classification": row.get("classification", "insufficient_data"),
                "confidence": float(row.get("confidence", 0)),
                "explanation": row.get("explanation", ""),
                "evidence_strength": row.get("evidence_strength", "no_evidence"),
                "evidence_sources": row.get("evidence_source", []),
                "source": row.get("source", "llm_inferred"),
            },
        }

        if timeseries_df is not None and start is not None and end is not None:
            ind_id = item["indicator_code"]
            geo_id = item["geography_code"]
            period_col = "period" if "period" in timeseries_df.columns else "YEAR"
            ind_col = "indicator_id" if "indicator_id" in timeseries_df.columns else "INDICATOR"
            geo_col = "geography_id" if "geography_id" in timeseries_df.columns else "REF_AREA"
            sub = timeseries_df[
                (timeseries_df[ind_col] == ind_id)
                & (timeseries_df[geo_col] == geo_id)
                & (timeseries_df[period_col] >= start)
                & (timeseries_df[period_col] <= end)
            ]
            value_col = "value" if "value" in timeseries_df.columns else "VALUE"
            imp_col = "is_imputed" if "is_imputed" in timeseries_df.columns else "Imputed"
            item["timeseries"] = [
                {
                    "period": int(r[period_col]),
                    "value": float(r[value_col]) if pd.notna(r[value_col]) else None,
                    "is_imputed": bool(r.get(imp_col, False)),
                }
                for _, r in sub.iterrows()
            ]
            item["timeseries"].sort(key=lambda x: x["period"])

        items.append(item)

    return items


def export_for_review(
    anomalies_df: pd.DataFrame,
    timeseries_df: Optional[pd.DataFrame] = None,
    output_path: str | Path = "anomaly_review.json",
) -> Path:
    """Export anomalies to JSON for reviewer app consumption.

    Parameters
    ----------
    anomalies_df : pd.DataFrame
        Anomaly explanations.
    timeseries_df : pd.DataFrame, optional
        Canonical timeseries (adds timeseries to each item).
    output_path : str or Path
        Output file path.

    Returns
    -------
    Path
        Path to written file.
    """
    output_path = Path(output_path)
    items = to_review_format(anomalies_df, timeseries_df)
    payload = {"version": "1.0", "count": len(items), "items": items}
    output_path.write_text(
        __import__("json").dumps(payload, indent=2, default=str),
        encoding="utf-8",
    )
    return output_path
