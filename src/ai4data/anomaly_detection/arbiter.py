"""Arbiter/judge LLM to harmonize multiple explainer outputs.

Compares explanations from different LLMs for the same anomaly and produces
a single, most-compelling explanation (or a combined one when both are valid).
"""

import json
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

import pandas as pd

from ai4data.anomaly_detection.schemas import AnomalyExplanation

ARBITER_SYSTEM_PROMPT = """You are a judge that evaluates multiple explanations for the same timeseries anomaly.

Given 2+ explanations from different LLMs for the same (indicator, geography, time window), your task is to:
1. Identify which explanation is most compelling (best supported by evidence).
2. If both explanations are valid and complementary, produce a combined explanation.
3. If neither is convincing, prefer "insufficient_data" classification.
4. Output ONLY valid JSON matching the schema—no markdown or extra text.

Be conservative: prefer existing explanations over inventing new ones."""

ARBITER_USER_TEMPLATE = """# ANOMALY CONTEXT
Indicator: {{indicator}}
Geography: {{geography}}
Window: {{window_str}}

# EXPLANATIONS TO COMPARE
{{explanations_json}}

# TASK
Pick the best explanation or combine valid ones. Output a single anomaly object (wrap in {"anomalies": [your_anomaly]}):
- window: [start_year, end_year]
- is_anomaly: bool
- classification: one of data_error | external_driver | measurement_system_update | modeling_artifact | insufficient_data
- confidence: 0-1
- explanation: your chosen/combined explanation text
- evidence_strength: strong_direct | moderate_contextual | weak_speculative | no_evidence
- evidence_source: list of {name, date_range, source_type, verifiability}
- source: "arbiter_harmonized"
"""


def _anomaly_to_dict(a: Dict[str, Any]) -> Dict[str, Any]:
    """Flatten anomaly for display (handle nested evidence_source)."""
    out = {k: v for k, v in a.items() if k != "evidence_source"}
    out["evidence_source"] = a.get("evidence_source", [])
    return out


def build_arbiter_payload(
    indicator: str,
    geography: str,
    window_str: str,
    explanations: List[Dict[str, Any]],
    system_prompt: str = ARBITER_SYSTEM_PROMPT,
    user_template: str = ARBITER_USER_TEMPLATE,
) -> Dict[str, Any]:
    """Build the payload for the arbiter LLM call.

    Parameters
    ----------
    indicator : str
        Indicator name.
    geography : str
        Geography name.
    window_str : str
        Window as "start-end".
    explanations : list of dict
        List of anomaly dicts from different explainers.
    system_prompt, user_template : str
        Override default prompts.

    Returns
    -------
    dict
        Messages for the arbiter call (system + user).
    """
    from jinja2 import Template

    explanations_json = json.dumps(
        [_anomaly_to_dict(e) for e in explanations],
        indent=2,
    )
    user_prompt = Template(user_template).render(
        indicator=indicator,
        geography=geography,
        window_str=window_str,
        explanations_json=explanations_json,
    )
    return {
        "system": system_prompt,
        "user": user_prompt,
    }


def group_explanations_by_context(
    dfs: List[pd.DataFrame],
    group_keys: tuple = ("indicator_code", "country_code", "window_str"),
) -> Dict[tuple, List[Dict[str, Any]]]:
    """Group anomaly rows from multiple DataFrames by (indicator, geography, window).

    Parameters
    ----------
    dfs : list of DataFrame
        One DataFrame per explainer (same schema).
    group_keys : tuple
        Columns to use as group key.

    Returns
    -------
    dict
        (indicator_code, country_code, window_str) -> list of anomaly dicts.
    """
    grouped: Dict[tuple, List[Dict[str, Any]]] = {}
    for df in dfs:
        if df.empty:
            continue
        for key, grp in df.groupby(list(group_keys)):
            if key not in grouped:
                grouped[key] = []
            for _, row in grp.iterrows():
                anomaly = row.to_dict()
                grouped[key].append(anomaly)
    return grouped


def harmonize_explanations(
    dfs: List[pd.DataFrame],
    invoke_llm: Callable[[str, str, List[Dict]], Dict[str, Any]],
    response_format: Optional[dict] = None,
) -> pd.DataFrame:
    """Harmonize multiple explainer outputs via an arbiter LLM.

    Parameters
    ----------
    dfs : list of DataFrame
        One DataFrame per explainer (same schema).
    invoke_llm : callable
        (system_prompt, user_prompt, response_format) -> parsed content dict.
        Should return {"anomalies": [anomaly_dict]} or a single anomaly dict.
    response_format : dict, optional
        JSON schema for structured output. If None, uses get_anomaly_response_format().

    Returns
    -------
    pd.DataFrame
        Harmonized anomalies, one row per (indicator, geography, window).
    """
    if response_format is None:
        from ai4data.anomaly_detection.prompts import get_anomaly_response_format

        response_format = get_anomaly_response_format()

    grouped = group_explanations_by_context(dfs)
    results = []

    for (indicator_code, country_code, window_str), explanations in grouped.items():
        if len(explanations) < 2:
            # Single explanation: keep as is, mark source
            r = explanations[0].copy()
            r["source"] = "single_explainer"
            results.append(r)
            continue

        indicator = explanations[0].get("indicator", indicator_code)
        geography = explanations[0].get("country", country_code)
        payload = build_arbiter_payload(
            indicator, geography, window_str, explanations
        )
        content = invoke_llm(
            payload["system"],
            payload["user"],
            response_format,
        )
        if content is None:
            # Fallback: use first explanation
            r = explanations[0].copy()
            r["source"] = "arbiter_fallback"
            results.append(r)
            continue

        anomalies = content.get("anomalies", [])
        if not anomalies:
            anomalies = [content] if isinstance(content, dict) and "window" in content else []

        for a in anomalies:
            r = {**explanations[0], **a}
            r["indicator_code"] = indicator_code
            r["country_code"] = country_code
            r["indicator"] = indicator
            r["country"] = geography
            r["window_str"] = window_str
            r["source"] = "arbiter_harmonized"
            results.append(r)

    if not results:
        return pd.DataFrame()
    return pd.DataFrame(results)
