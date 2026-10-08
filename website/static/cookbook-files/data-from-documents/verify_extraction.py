"""Verify data extracted from a document against the document itself.

Reads an extraction JSON (a chart snapshot or a table, see the example
files) and runs the checks that the document makes possible:

  chart   each extracted value is matched to a value label printed on the
          chart (the `printed_values` list, read from the snapshot image);
          charts without printed labels are reported as estimated, which
          this check cannot verify
  table   each numeric column sums to the total row within a tolerance;
          each derived column (a rate computed from two other columns)
          matches the computed value within a tolerance

Every value gets a status: verified, flagged, or estimated. Exit status is
0 when nothing is flagged, 1 otherwise. Standard library only.

What this does not check: whether the extraction picked the right table or
chart, whether the row and column labels are correct, or whether a value
without a printed label, a total, or a derivation is right. Those need a
person with the page open; recipe 4.2 samples them.

Usage:
    python verify_extraction.py example_snapshot.json
    python verify_extraction.py example_table.json --abs-tolerance 0.5 --rel-tolerance 0.02
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def close(a: float, b: float, abs_tol: float, rel_tol: float) -> bool:
    return abs(a - b) <= max(abs_tol, rel_tol * abs(b))


def verify_chart(x: dict) -> list[tuple[str, str, str]]:
    """Return (location, status, detail) per value."""
    results = []
    if not x.get("value_labels_printed"):
        for s in x["series"]:
            for cat, v in zip(x["categories"], s["values"]):
                results.append(
                    (f"{s['name']}/{cat}", "estimated", f"{v}: no printed label; read from the axis")
                )
        return results
    remaining = list(x.get("printed_values", []))
    for s in x["series"]:
        for cat, v in zip(x["categories"], s["values"]):
            if v in remaining:
                remaining.remove(v)
                results.append((f"{s['name']}/{cat}", "verified", f"{v} matches a printed label"))
            else:
                results.append((f"{s['name']}/{cat}", "flagged", f"{v} is not among the printed labels"))
    for leftover in remaining:
        results.append(("printed label", "flagged", f"{leftover} printed on the chart but not extracted"))
    return results


def verify_table(x: dict, abs_tol: float, rel_tol: float) -> list[tuple[str, str, str]]:
    results = []
    cols = x["columns"]
    rows = x["rows"]
    total_label = x.get("total_row")
    total = next((r for r in rows if r[0] == total_label), None)
    body = [r for r in rows if r[0] != total_label]
    derived = x.get("derived", {})
    for j, col in enumerate(cols[1:], start=1):
        values = [r[j] for r in rows if isinstance(r[j], (int, float))]
        if len(values) != len(rows):
            continue
        if col in derived:
            spec = derived[col]
            ni, di = cols.index(spec["numerator"]), cols.index(spec["denominator"])
            for r in rows:
                computed = r[ni] / r[di] * spec.get("scale", 1)
                status = "verified" if close(r[j], computed, abs_tol, rel_tol) else "flagged"
                results.append((f"{r[0]}/{col}", status, f"{r[j]} vs computed {computed:.2f}"))
        elif total is not None:
            s = sum(r[j] for r in body)
            status = "verified" if close(total[j], s, abs_tol, rel_tol) else "flagged"
            results.append((f"{total_label}/{col}", status, f"total {total[j]} vs sum of rows {s:g}"))
            for r in body:
                results.append(
                    (
                        f"{r[0]}/{col}",
                        "verified" if status == "verified" else "estimated",
                        f"{r[j]} (covered by the total check)",
                    )
                )
        else:
            for r in rows:
                results.append(
                    (f"{r[0]}/{col}", "estimated", f"{r[j]}: no total or derivation to check against")
                )
    return results


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("extraction", type=Path, help="chart snapshot or table JSON")
    parser.add_argument("--abs-tolerance", type=float, default=0.5)
    parser.add_argument("--rel-tolerance", type=float, default=0.02)
    args = parser.parse_args(argv)

    with args.extraction.open(encoding="utf-8") as fh:
        x = json.load(fh)
    kind = x.get("class", "").lower()
    if kind == "figure":
        results = verify_chart(x)
    elif kind == "table":
        results = verify_table(x, args.abs_tolerance, args.rel_tolerance)
    else:
        sys.exit("the extraction must have class Figure or Table")

    print(f"{x['document_id']} page {x['page']}: {x.get('title', '')} ({kind})")
    for where, status, detail in results:
        print(f"  {status:<9} {where:<32} {detail}")
    counts = {s: sum(1 for _, st, _ in results if st == s) for s in ("verified", "flagged", "estimated")}
    print(f"\n{counts['verified']} verified, {counts['flagged']} flagged, {counts['estimated']} estimated")
    print("Not checked here: whether the right table or chart was extracted, or whether labels are correct.")
    return 1 if counts["flagged"] else 0


if __name__ == "__main__":
    sys.exit(main())
