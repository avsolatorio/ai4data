"""Parse LLM batch output into structured anomaly DataFrames."""

import json
from pathlib import Path
from typing import Dict, List, Any

import pandas as pd

from ai4data.anomaly.explanation.explainers import (
    get_explainer,
    list_explainers,
    register_explainer,
)


def parse_batch_output(
    output_path: str | Path,
    provider: str,
    indicator_name_map: Dict[str, str],
    geography_name_map: Dict[str, str],
    custom_id_separator: str = "-",
    custom_id_parts: tuple = (0, 2, 3),  # (prefix, indicator_idx, geography_idx)
) -> pd.DataFrame:
    """Parse JSONL batch output into an anomaly explanation DataFrame.

    Parameters
    ----------
    output_path : str or Path
        Path to the JSONL file (OpenAI or Gemini batch output format).
    provider : str
        One of "openai" or "gemini".
    indicator_name_map : dict
        Maps indicator_id -> indicator_name.
    geography_name_map : dict
        Maps geography_id -> geography_name.
    custom_id_separator : str
        Separator in custom_id (e.g., "nosearch-c660ac92-INDICATOR-GEO-hash").
    custom_id_parts : tuple
        Indices for (prefix, indicator_code, country_code) in split(custom_id_separator).

    Returns
    -------
    pd.DataFrame
        One row per anomaly with columns: custom_id, indicator_code, indicator,
        country_code, country, window, is_anomaly, classification, confidence,
        explanation, evidence_strength, evidence_source, source, window_str.
    """
    parser = get_explainer(provider)
    if parser is None:
        raise ValueError(
            f"Unknown provider '{provider}'. "
            f"Available: {list_explainers()}. "
            "Use register_explainer() to add custom providers."
        )

    out_df = pd.read_json(output_path, lines=True)
    anomalies: List[Dict[str, Any]] = []

    for _, row in out_df.iterrows():
        row = row.copy()
        if provider == "gemini":
            row["custom_id"] = row.get("key", row.get("custom_id", ""))

        custom_id = row["custom_id"]
        parts = custom_id.split(custom_id_separator)
        if len(parts) < max(custom_id_parts) + 1:
            continue
        indicator_code = parts[custom_id_parts[1]]
        country_code = parts[custom_id_parts[2]]

        content = parser(row)
        if content is None:
            continue

        for anomaly in content.get("anomalies", []):
            anomalies.append({
                "custom_id": custom_id,
                "indicator_code": indicator_code,
                "indicator": indicator_name_map.get(indicator_code, indicator_code),
                "country_code": country_code,
                "country": geography_name_map.get(country_code, country_code),
                **anomaly,
            })

    if not anomalies:
        return pd.DataFrame()

    df = pd.DataFrame(anomalies)
    df["window_str"] = df["window"].apply(
        lambda x: f"{x[0]}-{x[1]}" if isinstance(x, (list, tuple)) and len(x) >= 2 else ""
    )
    return df
