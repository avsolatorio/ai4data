"""Convert an extracted chart or table into a tidy CSV with provenance.

One row per value, with the columns that let any later user trace the
value back to the page: document_id, page, bbox, class, title, row,
column, value, unit, status. The status column records the verification
result when a verification report from verify_extraction.py is given;
otherwise every value is "extracted".

Standard library only.

Usage:
    python snapshot_to_tidy.py example_table.json -o T3.1.csv
    python verify_extraction.py example_table.json > report.txt
    python snapshot_to_tidy.py example_table.json --verification report.txt -o T3.1.csv

What this does not do: it does not change values. A flagged value stays in
the file with status "flagged" so that the correction is a documented
edit, with the page open.
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import sys
from pathlib import Path

REPORT_LINE = re.compile(r"^\s+(verified|flagged|estimated)\s+(\S.*?)\s{2,}")


def read_statuses(report: Path | None) -> dict[str, str]:
    if report is None:
        return {}
    statuses = {}
    for line in report.read_text(encoding="utf-8").splitlines():
        m = REPORT_LINE.match(line)
        if m:
            statuses[m.group(2).strip()] = m.group(1)
    return statuses


def rows_for(x: dict, statuses: dict[str, str]) -> list[dict]:
    base = {
        "document_id": x["document_id"],
        "page": x["page"],
        "bbox": " ".join(str(b) for b in x["bbox"]),
        "class": x["class"],
        "title": x.get("title", ""),
    }
    out = []
    if x["class"].lower() == "figure":
        for s in x["series"]:
            for cat, v in zip(x["categories"], s["values"]):
                key = f"{s['name']}/{cat}"
                out.append(
                    {
                        **base,
                        "row": cat,
                        "column": s["name"],
                        "value": v,
                        "unit": x.get("unit", ""),
                        "status": statuses.get(key, "extracted"),
                    }
                )
    else:
        cols = x["columns"]
        units = x.get("units", {})
        for r in x["rows"]:
            for j, col in enumerate(cols[1:], start=1):
                key = f"{r[0]}/{col}"
                out.append(
                    {
                        **base,
                        "row": r[0],
                        "column": col,
                        "value": r[j],
                        "unit": units.get(col, ""),
                        "status": statuses.get(key, "extracted"),
                    }
                )
    return out


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("extraction", type=Path)
    parser.add_argument("--verification", type=Path, help="output of verify_extraction.py")
    parser.add_argument("-o", "--output", type=Path)
    args = parser.parse_args(argv)

    with args.extraction.open(encoding="utf-8") as fh:
        x = json.load(fh)
    rows = rows_for(x, read_statuses(args.verification))
    fields = ["document_id", "page", "bbox", "class", "title", "row", "column", "value", "unit", "status"]
    out = args.output.open("w", newline="", encoding="utf-8") if args.output else sys.stdout
    writer = csv.DictWriter(out, fieldnames=fields)
    writer.writeheader()
    writer.writerows(rows)
    if args.output:
        out.close()
        print(f"wrote {args.output}: {len(rows)} values")
    return 0


if __name__ == "__main__":
    sys.exit(main())
