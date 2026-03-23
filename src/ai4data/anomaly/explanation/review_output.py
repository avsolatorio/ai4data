"""Reviewer-friendly output format for anomaly explanation pipeline.

Designed for easy integration into review applications: combines explanations
with timeseries data for display.
"""

import json
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

import pandas as pd

from ai4data.anomaly.explanation.arbiter import (
    build_arbiter_payload,
    group_explanations_by_context_with_providers,
)


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


# Default padding (years before/after anomaly window) so reviewer sees surrounding context
DEFAULT_PERIOD_PADDING = 5


def to_review_format(
    anomalies_df: pd.DataFrame,
    timeseries_df: Optional[pd.DataFrame] = None,
    indicator_col: str = "indicator_code",
    geography_col: str = "country_code",
    window_col: str = "window",
    period_padding: int = DEFAULT_PERIOD_PADDING,
) -> List[Dict[str, Any]]:
    """Convert anomaly DataFrame to review-app-friendly format.

    Parameters
    ----------
    anomalies_df : pd.DataFrame
        Output from parse_batch_output or harmonize_explanations.
    timeseries_df : pd.DataFrame, optional
        Canonical long-format timeseries. If provided, each item includes
        a "timeseries" array with data around the anomaly window (extended by
        period_padding years before/after so the reviewer sees full context).
    indicator_col, geography_col, window_col : str
        Column names for indicator, geography, and window.
    period_padding : int
        Years to include before and after the anomaly window (default 5).

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
            item["timeseries"] = _extract_timeseries(
                timeseries_df,
                item["indicator_code"],
                item["geography_code"],
                start,
                end,
                period_padding=period_padding,
            )

        items.append(item)

    return items


def _extract_timeseries(
    timeseries_df: pd.DataFrame,
    indicator_code: str,
    geography_code: str,
    start: Optional[int],
    end: Optional[int],
    period_padding: int = DEFAULT_PERIOD_PADDING,
) -> List[Dict[str, Any]]:
    """Extract timeseries for an anomaly, including padding around the anomaly window.

    Includes period_padding years before and after the anomaly so the reviewer
    sees the full context (the anomaly window is still used for chart highlighting).
    """
    if timeseries_df is None or start is None or end is None:
        return []
    period_col = "period" if "period" in timeseries_df.columns else "YEAR"
    ind_col = "indicator_id" if "indicator_id" in timeseries_df.columns else "INDICATOR"
    geo_col = "geography_id" if "geography_id" in timeseries_df.columns else "REF_AREA"

    base = timeseries_df[
        (timeseries_df[ind_col] == indicator_code)
        & (timeseries_df[geo_col] == geography_code)
    ]
    if base.empty:
        return []
    min_period = int(base[period_col].min())
    max_period = int(base[period_col].max())
    extended_start = max(min_period, start - period_padding)
    extended_end = min(max_period, end + period_padding)

    sub = base[
        (base[period_col] >= extended_start)
        & (base[period_col] <= extended_end)
    ]
    value_col = "value" if "value" in timeseries_df.columns else "VALUE"
    imp_col = "is_imputed" if "is_imputed" in timeseries_df.columns else "Imputed"
    items = [
        {
            "period": int(r[period_col]),
            "value": float(r[value_col]) if pd.notna(r.get(value_col)) else None,
            "is_imputed": bool(r.get(imp_col, False)),
        }
        for _, r in sub.iterrows()
    ]
    items.sort(key=lambda x: x["period"])
    return items


def _anomaly_to_explainer_row(provider_name: str, anomaly: Dict[str, Any]) -> Dict[str, Any]:
    """Convert anomaly dict to explainer entry for review format."""
    ev = anomaly.get("evidence_source", anomaly.get("evidence_sources", []))
    ev_list = ev if isinstance(ev, list) else []
    return {
        "name": provider_name,
        "classification": str(anomaly.get("classification", "insufficient_data")),
        "explanation": str(anomaly.get("explanation", "")),
        "evidence_sources": [e if isinstance(e, dict) else {"name": str(e)} for e in ev_list],
        "confidence": float(anomaly.get("confidence", 0)),
    }


def to_review_format_with_explainers(
    grouped: Dict[tuple, List[Tuple[str, Dict[str, Any]]]],
    timeseries_df: Optional[pd.DataFrame] = None,
    indicator_col: str = "indicator_code",
    geography_col: str = "country_code",
    window_col: str = "window",
    period_padding: int = DEFAULT_PERIOD_PADDING,
    run_arbiter: bool = False,
    invoke_llm: Optional[Callable[..., Optional[Dict[str, Any]]]] = None,
    response_format: Optional[dict] = None,
) -> List[Dict[str, Any]]:
    """Convert grouped explainer outputs to review format with explanation.explainers.

    Parameters
    ----------
    grouped : dict
        From group_explanations_by_context_with_providers: (ind, geo, window_str) -> [(provider, anomaly_dict), ...].
    timeseries_df : pd.DataFrame, optional
        Canonical long-format timeseries.
    indicator_col, geography_col, window_col : str
        Column names for grouping key mapping.
    run_arbiter : bool
        If True and multiple explainers, run arbiter for primary classification.
    invoke_llm : callable, optional
        (system_prompt, user_prompt, response_format) -> parsed content. Required if run_arbiter.
    response_format : dict, optional
        JSON schema for arbiter. Uses get_anomaly_response_format() if None.

    Returns
    -------
    list of dict
        Items in review format with explanation.explainers for tabbed UI.
    """
    if response_format is None and run_arbiter:
        from ai4data.anomaly.explanation.prompts import get_anomaly_response_format
        response_format = get_anomaly_response_format()

    group_keys = ("indicator_code", "country_code", "window_str")
    items: List[Dict[str, Any]] = []

    for key, provider_anomalies in grouped.items():
        if not provider_anomalies:
            continue
        ind_code = key[0] if len(key) >= 1 else ""
        geo_code = key[1] if len(key) >= 2 else ""
        window_str = key[2] if len(key) >= 3 else ""
        first = provider_anomalies[0][1]
        window = first.get(window_col, first.get("window", []))
        if isinstance(window, (list, tuple)) and len(window) >= 2:
            start, end = int(window[0]), int(window[1])
        else:
            start = end = None

        explainers_list = [_anomaly_to_explainer_row(p, a) for p, a in provider_anomalies]
        classifications = [e["classification"] for e in explainers_list]
        agreement = "agree" if len(set(classifications)) <= 1 and classifications else "disagree"
        if len(provider_anomalies) == 1:
            agreement = "single"
        primary_classification = classifications[0] if classifications else "insufficient_data"

        if run_arbiter and len(provider_anomalies) >= 2 and invoke_llm:
            indicator = first.get("indicator", ind_code)
            geography = first.get("country", geo_code)
            payload = build_arbiter_payload(
                indicator, geography, window_str,
                [a for _, a in provider_anomalies],
            )
            content = invoke_llm(
                payload["system"],
                payload["user"],
                response_format,
            )
            if content:
                anomalies = content.get("anomalies", [])
                if not anomalies and isinstance(content, dict) and "window" in content:
                    anomalies = [content]
                if anomalies:
                    a = anomalies[0]
                    primary_classification = str(a.get("classification", primary_classification))

        item = {
            "indicator_code": ind_code,
            "indicator_name": first.get("indicator", first.get("indicator_name", ind_code)),
            "geography_code": geo_code,
            "geography_name": first.get("country", first.get("geography_name", geo_code)),
            "window_start": start,
            "window_end": end,
            "window_str": window_str,
            "timeseries": _extract_timeseries(
                timeseries_df, ind_code, geo_code, start, end,
                period_padding=period_padding,
            ),
            "explanation": {
                "is_anomaly": first.get("is_anomaly", True),
                "classification": primary_classification,
                "agreement": agreement,
                "explainers": explainers_list,
                "confidence": float(first.get("confidence", 0.85)),
                "evidence_strength": first.get("evidence_strength", "no_evidence"),
                "evidence_sources": [],
                "source": "pipeline",
            },
        }
        items.append(item)

    return items


def export_for_review_with_explainers(
    explainer_dfs: List[Tuple[str, pd.DataFrame]],
    timeseries_df: Optional[pd.DataFrame] = None,
    output_path: str | Path = "anomaly_review.json",
    period_padding: int = DEFAULT_PERIOD_PADDING,
    run_arbiter: bool = False,
    invoke_llm: Optional[Callable[..., Optional[Dict[str, Any]]]] = None,
) -> Path:
    """Export multi-explainer outputs to JSON with explanation.explainers for tabbed UI.

    Parameters
    ----------
    explainer_dfs : list of (provider_name, DataFrame)
        e.g. [("OpenAI", df_openai), ("Gemini", df_gemini)].
        Each DataFrame is output from parse_batch_output for that provider.
    timeseries_df : pd.DataFrame, optional
        Canonical long-format timeseries.
    output_path : str or Path
        Output file path.
    period_padding : int
        Years to include before/after anomaly window in timeseries (default 5).
    run_arbiter : bool
        If True and multiple explainers, run arbiter for primary classification.
    invoke_llm : callable, optional
        Required if run_arbiter. (system_prompt, user_prompt, response_format) -> parsed dict.

    Returns
    -------
    Path
        Path to written file.
    """
    grouped = group_explanations_by_context_with_providers(explainer_dfs)
    items = to_review_format_with_explainers(
        grouped,
        timeseries_df=timeseries_df,
        period_padding=period_padding,
        run_arbiter=run_arbiter,
        invoke_llm=invoke_llm,
    )
    out = Path(output_path)
    payload = {"version": "1.0", "count": len(items), "items": items}
    out.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
    return out


def export_for_review(
    anomalies_df: pd.DataFrame,
    timeseries_df: Optional[pd.DataFrame] = None,
    output_path: str | Path = "anomaly_review.json",
    period_padding: int = DEFAULT_PERIOD_PADDING,
) -> Path:
    """Export anomalies to JSON for reviewer app consumption.

    Parameters
    ----------
    anomalies_df : pd.DataFrame
        Anomaly explanations.
    timeseries_df : pd.DataFrame, optional
        Canonical timeseries (adds timeseries to each item, with context around
        the anomaly window).
    output_path : str or Path
        Output file path.
    period_padding : int
        Years to include before/after anomaly window in timeseries (default 5).

    Returns
    -------
    Path
        Path to written file.
    """
    output_path = Path(output_path)
    items = to_review_format(
        anomalies_df,
        timeseries_df,
        period_padding=period_padding,
    )
    payload = {"version": "1.0", "count": len(items), "items": items}
    output_path.write_text(
        __import__("json").dumps(payload, indent=2, default=str),
        encoding="utf-8",
    )
    return output_path
