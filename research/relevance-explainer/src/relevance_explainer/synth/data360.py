"""Data360 search client and record compaction."""
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
