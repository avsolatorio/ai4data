#!/usr/bin/env python3
"""Convert overlap_explanations.csv (original pipeline output) to app review JSON format.

When --timeseries is provided (path to canonical long-format CSV or Excel), uses
actual indicator values. Otherwise generates synthetic placeholder data.
"""

import argparse
import ast
import json
import re
from pathlib import Path
from typing import Optional

import pandas as pd


def parse_window_str(window_str: str) -> tuple[int, int]:
    """Parse 'YYYY-YYYY' or 'YYYY' to (start, end)."""
    if not window_str or pd.isna(window_str):
        return (0, 0)
    parts = str(window_str).strip().split("-")
    if len(parts) == 1:
        y = int(parts[0])
        return (y, y)
    return (int(parts[0]), int(parts[1]))


def extract_indicator_from_custom_id(custom_id: str) -> str:
    """Extract indicator code from custom_id like nosearch-*-WB_CSC_EG_ELC_ACCS_ZS-ATG-*."""
    if not custom_id or pd.isna(custom_id):
        return ""
    parts = str(custom_id).split("-")
    # Pattern: nosearch-{hash}-{indicator_code}-{country_code}-{hash}
    if len(parts) >= 4:
        return parts[2]
    return ""


def get_country_name(code: str) -> str:
    """Get country/region name from alpha-3 code."""
    try:
        import pycountry

        c = pycountry.countries.get(alpha_3=code)
        if c:
            return c.name
        # Try as subdivision (e.g. regions)
        for subdiv in pycountry.subdivisions:
            if subdiv.code == code:
                return subdiv.name
    except Exception:
        pass
    # Fallback for common region codes
    REGIONS = {
        "BEC": "Europe & Central Asia (IBRD only)",
        "BFA": "Burkina Faso",
        "DEA": "East Asia & Pacific (IDA total)",
        "DLA": "Latin America & Caribbean (IDA total)",
        "IBD": "IBRD countries",
        "IBT": "IDA & IBRD total",
        "IDA": "IDA countries",
        "SID": "Small Island Developing States",
        "TSA": "South Asia",
    }
    return REGIONS.get(code.upper() if code else "", code or "")


def safe_parse_evidence(s: str) -> list:
    """Parse evidence_source string (Python repr of list of dicts) to list."""
    if not s or pd.isna(s) or str(s).strip() in ("[]", ""):
        return []
    raw = str(s).strip()
    try:
        # Handle single-quoted dicts (Python repr)
        parsed = ast.literal_eval(raw.replace("'", '"'))
        return parsed if isinstance(parsed, list) else []
    except Exception:
        return []


def format_evidence(ev_list: list) -> str:
    """Format evidence list as readable text."""
    if not ev_list:
        return ""
    lines = []
    for i, ev in enumerate(ev_list, 1):
        if isinstance(ev, dict):
            name = ev.get("name", ev.get("source", ""))
            dr = ev.get("date_range", "")
            st = ev.get("source_type", "")
            parts = [name]
            if dr:
                parts.append(f"({dr})")
            if st:
                parts.append(st)
            lines.append(f"  {i}. {' '.join(parts)}")
        else:
            lines.append(f"  {i}. {ev}")
    return "\n".join(lines) if lines else ""


# Match DEFAULT_PERIOD_PADDING in review_output so reviewer sees full context
PERIOD_PADDING = 5


def load_timeseries(path: Path) -> Optional[pd.DataFrame]:
    """Load timeseries from CSV or Excel into canonical format."""
    if not path.exists():
        return None
    suffix = path.suffix.lower()
    if suffix == ".csv":
        df = pd.read_csv(path)
    elif suffix in (".xlsx", ".xls"):
        df = pd.read_excel(path)
    else:
        return None
    # Normalize to canonical columns
    col_map = {
        "INDICATOR": "indicator_id",
        "REF_AREA": "geography_id",
        "YEAR": "period",
        "VALUE": "value",
        "Imputed": "is_imputed",
    }
    for old, new in col_map.items():
        if old in df.columns and new not in df.columns:
            df = df.rename(columns={old: new})
    if "indicator_id" not in df.columns and "indicator_code" in df.columns:
        df = df.rename(columns={"indicator_code": "indicator_id"})
    if "geography_id" not in df.columns and "country_code" in df.columns:
        df = df.rename(columns={"country_code": "geography_id"})
    return df


def synthetic_timeseries(window_start: int, window_end: int) -> list[dict]:
    """Generate placeholder timeseries for chart display (CSV has no actual values)."""
    pad = PERIOD_PADDING
    start = max(1980, window_start - pad)
    end = min(2030, window_end + pad)
    periods = list(range(start, end + 1))
    # Create a simple pattern: baseline, dip/spike in window, return
    values = []
    for p in periods:
        if p < window_start:
            values.append(50.0 + (p - start) * 2)
        elif window_start <= p <= window_end:
            values.append(40.0 + (p - window_start) * 5)
        else:
            values.append(50.0 + (p - window_end) * 2)
    return [
        {"period": int(p), "value": round(v, 2), "is_imputed": False}
        for p, v in zip(periods, values)
    ]


def row_to_item(
    row: pd.Series,
    timeseries_df: Optional[pd.DataFrame] = None,
) -> dict:
    """Convert a CSV row to app item format."""
    custom_id = row.get("custom_id", "")
    indicator_name = row.get("indicator", "")
    country_code = row.get("country_code", "")
    window_str = str(row.get("window_str", "")).strip()
    window_start, window_end = parse_window_str(window_str)

    indicator_code = extract_indicator_from_custom_id(custom_id)
    if not indicator_code:
        indicator_code = re.sub(r"[^A-Za-z0-9_]", "", indicator_name)[:50]

    geography_name = get_country_name(str(country_code).strip())

    expl_openai = str(row.get("explanation_openai", "")).strip() if pd.notna(row.get("explanation_openai")) else ""
    expl_gemini = str(row.get("explanation_gemini", "")).strip() if pd.notna(row.get("explanation_gemini")) else ""
    cl_openai = str(row.get("classification_openai", "")).strip() if pd.notna(row.get("classification_openai")) else ""
    cl_gemini = str(row.get("classification_gemini", "")).strip() if pd.notna(row.get("classification_gemini")) else ""

    ev_openai = safe_parse_evidence(row.get("evidence_source_openai", ""))
    ev_gemini = safe_parse_evidence(row.get("evidence_source_gemini", ""))

    # Build explainers array for tabbed UI
    explainers = []
    if expl_openai:
        explainers.append({
            "name": "OpenAI",
            "classification": cl_openai or "insufficient_data",
            "explanation": expl_openai,
            "evidence_sources": ev_openai,
            "confidence": 0.85,
        })
    if expl_gemini:
        explainers.append({
            "name": "Gemini",
            "classification": cl_gemini or "insufficient_data",
            "explanation": expl_gemini,
            "evidence_sources": ev_gemini,
            "confidence": 0.85,
        })

    # Primary classification and agreement
    classification = cl_openai or cl_gemini or "insufficient_data"
    agreement = "agree" if (cl_openai == cl_gemini and cl_openai) else "disagree" if (cl_openai and cl_gemini) else "single"

    # Use actual timeseries when available
    if timeseries_df is not None and window_start and window_end:
        try:
            from ai4data.anomaly_detection.review_output import _extract_timeseries

            ts = _extract_timeseries(
                timeseries_df,
                indicator_code,
                country_code,
                window_start,
                window_end,
                period_padding=PERIOD_PADDING,
            )
            if ts:
                timeseries = ts
            else:
                timeseries = synthetic_timeseries(window_start, window_end)
        except Exception:
            timeseries = synthetic_timeseries(window_start, window_end)
    else:
        timeseries = synthetic_timeseries(window_start, window_end)

    return {
        "indicator_code": indicator_code,
        "indicator_name": indicator_name,
        "geography_code": country_code,
        "geography_name": geography_name,
        "window_start": window_start,
        "window_end": window_end,
        "window_str": window_str,
        "timeseries": timeseries,
        "explanation": {
            "is_anomaly": True,
            "classification": classification,
            "agreement": agreement,
            "explainers": explainers,
            "confidence": 0.85,
            "evidence_strength": "multiple_sources" if (ev_openai or ev_gemini) else "no_evidence",
            "evidence_sources": ev_openai + ev_gemini,
            "source": "overlap_csv",
        },
    }


def main():
    parser = argparse.ArgumentParser(
        description="Convert overlap_explanations.csv to review app JSON"
    )
    parser.add_argument(
        "csv_path",
        type=Path,
        nargs="?",
        default=Path("overlap_explanations.csv"),
    )
    parser.add_argument("-o", "--output", type=Path, default=None)
    parser.add_argument(
        "-t",
        "--timeseries",
        type=Path,
        default=None,
        help="Path to canonical timeseries (CSV/Excel) for actual values",
    )
    args = parser.parse_args()

    csv_path = args.csv_path
    if not csv_path.exists():
        raise FileNotFoundError(f"CSV not found: {csv_path}")

    timeseries_df = load_timeseries(args.timeseries) if args.timeseries else None
    if timeseries_df is not None:
        print(f"Using actual timeseries from {args.timeseries}")
    else:
        print("Using synthetic timeseries (no --timeseries provided)")

    df = pd.read_csv(csv_path)
    items = [
        row_to_item(row, timeseries_df=timeseries_df)
        for _, row in df.iterrows()
    ]

    out = {
        "version": "1.0",
        "count": len(items),
        "items": items,
    }

    output_path = args.output or csv_path.with_name("overlap_review.json")
    output_path.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(f"Wrote {len(items)} items to {output_path}")
    print(f"Load in app: uv run python -m apps.anomaly_review {output_path}")


if __name__ == "__main__":
    main()
