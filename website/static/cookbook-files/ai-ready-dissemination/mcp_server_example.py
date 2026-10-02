"""A minimal MCP server over a statistics catalog.

Exposes two read-only tools to any MCP-compatible AI client: one to search
the catalog and one to fetch values for a series. The data come from the two
example CSV files next to this script; in production the tools would call
the office's existing API.

Requires the official Python SDK:  pip install "mcp[cli]"
Run locally:                       python mcp_server_example.py
Try it with the inspector:         mcp dev mcp_server_example.py
"""

import csv
from pathlib import Path

from mcp.server.fastmcp import FastMCP

HERE = Path(__file__).parent
mcp = FastMCP("statistics-example")


def _read(name):
    with open(HERE / name, newline="", encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


CATALOG = _read("example_catalog.csv")
VALUES = _read("example_values.csv")


@mcp.tool()
def search_series(query: str, limit: int = 5) -> list[dict]:
    """Search published series by words in the title or description.

    Returns id, title, unit, frequency, and coverage for each match.
    """
    words = query.lower().split()
    hits = []
    for rec in CATALOG:
        text = (rec["title"] + " " + rec["description"]).lower()
        score = sum(1 for w in words if w in text)
        if score:
            hits.append((score, rec))
    hits.sort(key=lambda h: h[0], reverse=True)
    return [
        {k: rec[k] for k in ("id", "title", "unit", "frequency", "start_period", "end_period")}
        for _, rec in hits[:limit]
    ]


@mcp.tool()
def get_values(series_id: str, geography: str = "NAT") -> dict:
    """Return the published values for a series.

    The response carries the unit, the release date, and the source URL so
    that the client can cite them.
    """
    rows = [v for v in VALUES if v["series_id"] == series_id and v["geography"] == geography]
    if not rows:
        return {"error": f"no values for {series_id} / {geography}"}
    return {
        "series_id": series_id,
        "geography": geography,
        "unit": rows[0]["unit"],
        "release": rows[0]["release"],
        "source_url": rows[0]["source_url"],
        "values": [{"period": r["period"], "value": float(r["value"])} for r in rows],
    }


if __name__ == "__main__":
    mcp.run()
