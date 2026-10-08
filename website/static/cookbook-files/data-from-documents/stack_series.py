"""Combine tidy extractions from several documents into one series.

Reads tidy CSV files (the output of snapshot_to_tidy.py: one row per value
with document_id, page, title, row, column, value, unit, status) and a
mapping file that says which table in which document feeds which series,
which of the table's dimensions is the area and which the period, and
which unit labels mean the same unit. Writes one series file with one row
per area and period, the value, the source document and page, and a
status:

    single      the value appears in one document
    confirmed   two documents report the same value for the area and period
    revised     a later document reports a different value; the later
                value is kept, the earlier one is recorded beside it

Documents are ordered by their year in the mapping's document identifier
order of appearance, earliest first. Prints the counts and the revisions.
Standard library only.

Usage:
    python stack_series.py series_mapping.csv tidy_fs2024.csv tidy_fs2025.csv -o series_fs.csv

What this does not do: it matches areas by exact label and periods by
exact string; a renamed region or a fiscal-year period needs a mapping
row. It keeps the latest value of a revision, which is right for a
revised statistic and wrong for a definitional change; the documents'
notes say which, and the series record carries the break.
"""

from __future__ import annotations

import argparse
import csv
import sys
from collections import Counter
from pathlib import Path


def load_mapping(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("mapping", type=Path)
    parser.add_argument("tidy", nargs="+", type=Path)
    parser.add_argument("-o", "--output", type=Path)
    args = parser.parse_args(argv)
    mapping = load_mapping(args.mapping)
    order = {m["document_id"]: i for i, m in enumerate(mapping)}

    rows: list[dict[str, str]] = []
    for path in args.tidy:
        with path.open(newline="", encoding="utf-8") as fh:
            rows.extend(csv.DictReader(fh))
    rows.sort(key=lambda r: order.get(r["document_id"], len(order)))

    series: dict[tuple[str, str, str], dict[str, str]] = {}
    unmatched = 0
    for r in rows:
        m = next(
            (
                m
                for m in mapping
                if m["document_id"] == r["document_id"]
                and r["title"].startswith(m["title_prefix"])
            ),
            None,
        )
        if m is None:
            unmatched += 1
            continue
        aliases = {a.strip().lower() for a in m["unit_aliases"].split(";")}
        if r["unit"].strip().lower() not in aliases:
            unmatched += 1
            continue
        area, period = (
            (r["row"], r["column"])
            if m["row_role"] == "area"
            else (r["column"], r["row"])
        )
        key = (m["series_id"], area, period)
        new = {
            "series_id": m["series_id"],
            "area": area,
            "period": period,
            "value": r["value"],
            "unit": m["unit_label"],
            "source_document": r["document_id"],
            "source_page": r["page"],
            "status": "single",
            "previous_value": "",
            "previous_document": "",
        }
        old = series.get(key)
        if old is None:
            series[key] = new
        elif old["value"] == new["value"]:
            old["status"] = "confirmed"
        else:
            new["status"] = "revised"
            new["previous_value"] = old["value"]
            new["previous_document"] = old["source_document"]
            series[key] = new

    out_rows = [series[k] for k in sorted(series)]
    fields = list(out_rows[0]) if out_rows else []
    out = (
        args.output.open("w", newline="", encoding="utf-8")
        if args.output
        else sys.stdout
    )
    writer = csv.DictWriter(out, fieldnames=fields)
    writer.writeheader()
    writer.writerows(out_rows)
    if args.output:
        out.close()
    counts = Counter(r["status"] for r in out_rows)
    print(
        f"{len(rows)} tidy rows from {len(args.tidy)} documents -> {len(out_rows)} series values; {unmatched} rows unmatched",
        file=sys.stderr,
    )
    print(
        ", ".join(f"{k} {counts[k]}" for k in ("single", "confirmed", "revised")),
        file=sys.stderr,
    )
    for r in out_rows:
        if r["status"] == "revised":
            print(
                f"revised: {r['area']} {r['period']}: {r['previous_value']} ({r['previous_document']}) -> {r['value']} ({r['source_document']})",
                file=sys.stderr,
            )
    return 0


if __name__ == "__main__":
    sys.exit(main())
