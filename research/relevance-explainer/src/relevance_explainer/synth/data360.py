"""Data360 search client, record compaction, filter checks, and prompt rendering."""
from __future__ import annotations

import json
import re
import urllib.request

URL = "https://data360api.worldbank.org/data360/searchv2"
# The API rejects Python's default user agent.
HEADERS = {"Content-Type": "application/json", "User-Agent": "curl/8.5.0"}


def search(query: str, top: int = 30, skip: int = 0) -> list[dict]:
    body = json.dumps({"search": query, "top": top, "skip": skip}).encode()
    req = urllib.request.Request(URL, body, HEADERS)
    with urllib.request.urlopen(req, timeout=60) as r:
        return json.load(r)["value"]


def _clip(s: str | None, n: int) -> str:
    s = re.sub(r"\s+", " ", s or "").strip()
    return s if len(s) <= n else s[: n - 1].rstrip() + "…"


def compact(raw: dict) -> dict | None:
    """Reduce a raw search hit to the metadata fields shown to the model."""
    sd = raw.get("series_description") or {}
    name = sd.get("name")
    if raw.get("type") != "indicator" or not name:
        return None
    years = [int(t[k]) for t in sd.get("time_periods") or [] for k in ("start", "end")
             if str(t.get(k) or "").isdigit()]
    countries = sorted({c["name"] for c in sd.get("ref_country") or [] if c.get("name")})
    return {
        "id": raw["id"].removeprefix("META_"),
        "name": name,
        "database": sd.get("database_name") or sd.get("database_id"),
        "source": next((a.get("name") for a in sd.get("authoring_entity") or [] if a.get("name")), None),
        "unit": sd.get("measurement_unit"),
        "frequency": sd.get("periodicity"),
        "definition": _clip(sd.get("definition_short") or sd.get("definition_long"), 450),
        "concept": _clip(sd.get("statistical_concept"), 250),
        "limitation": _clip(sd.get("limitation"), 200),
        "topics": [t["name"] for t in sd.get("topics") or [] if t.get("name")],
        "year_start": min(years) if years else None,
        "year_end": max(years) if years else None,
        "countries": countries,
    }


def _coverage_line(rec: dict) -> str:
    n = len(rec["countries"])
    if n == 0:
        return "not stated"
    if n <= 12:
        return f"{n} countries: " + ", ".join(rec["countries"])
    return f"{n} countries and economies"


def filter_check(rec: dict, filters: dict) -> list[str]:
    """Deterministic coverage checks for the filters that can be verified from metadata."""
    out = []
    for c in filters.get("countries") or []:
        if not rec["countries"]:
            out.append(f"country {c}: coverage not stated")
        else:
            ok = c.lower() in {x.lower() for x in rec["countries"]}
            out.append(f"country {c}: {'covered' if ok else 'not covered'}")
    yrs = filters.get("years")
    if yrs:
        m = re.fullmatch(r"\s*(\d{4})\s*(?:-|–|to)?\s*(\d{4})?\s*", str(yrs))
        if m and rec["year_start"] and rec["year_end"]:
            a, b = int(m.group(1)), int(m.group(2) or m.group(1))
            if rec["year_start"] <= a and b <= rec["year_end"]:
                status = "fully covered"
            elif b < rec["year_start"] or a > rec["year_end"]:
                status = "not covered"
            else:
                status = "partly covered"
            out.append(f"years {yrs}: {status}")
        else:
            out.append(f"years {yrs}: coverage not stated")
    freq = filters.get("frequency")
    if freq:
        if rec["frequency"]:
            same = freq.lower() == rec["frequency"].lower()
            out.append(f"frequency {freq}: {'matches' if same else 'differs (' + rec['frequency'] + ')'}")
        else:
            out.append(f"frequency {freq}: not stated")
    return out


def render_input(query: str, filters: dict, rec: dict, language: str = "English") -> str:
    """Text input shown to the teacher and, later, to the student model."""
    fl = "; ".join(f"{k}={'/'.join(v) if isinstance(v, list) else v}" for k, v in filters.items()) or "none"
    span = f"{rec['year_start']}-{rec['year_end']}" if rec["year_start"] else "not stated"
    lines = [
        f"Respond in: {language}",
        f"Query: {query}",
        f"Filters: {fl}",
        f"Result: {rec['name']}",
        f"Database: {rec['database'] or 'n/a'}; Source: {rec['source'] or 'n/a'}",
        f"Unit: {rec['unit'] or 'n/a'}; Frequency: {rec['frequency'] or 'n/a'}; Years: {span}",
        f"Coverage: {_coverage_line(rec)}",
        f"Topics: {', '.join(rec['topics']) or 'n/a'}",
        f"Definition: {rec['definition'] or 'n/a'}",
    ]
    if rec["concept"]:
        lines.append(f"Concept: {rec['concept']}")
    if rec["limitation"]:
        lines.append(f"Limitations: {rec['limitation']}")
    checks = filter_check(rec, filters)
    if checks:
        lines.append("Filter check: " + "; ".join(checks))
    return "\n".join(lines)
