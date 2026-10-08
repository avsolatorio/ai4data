"""Report on a document inventory: where the data are, and what is left to do.

Reads an inventory CSV with one row per document (document_id, title,
year, type, pages, text_layer, tables, figures, extracted, data_sources)
and prints counts by type and year, the number of tables and figures, the
share of documents with a text layer (documents without one need OCR
before any text-based extraction), and the extraction backlog ordered by
the number of tables and figures. Standard library only.

Usage:
    python inventory_report.py document_inventory.csv

What this does not do: it reports what the inventory says. Counting tables
and figures per document is the work of the layout detection in chapter 2;
until then the counts are estimates entered by hand.
"""

from __future__ import annotations

import argparse
import csv
import sys
from collections import Counter, defaultdict
from pathlib import Path


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("inventory", type=Path)
    args = parser.parse_args(argv)

    with args.inventory.open(newline="", encoding="utf-8") as fh:
        docs = list(csv.DictReader(fh))
    if not docs:
        sys.exit("no documents in the inventory")

    n = len(docs)
    tables = sum(int(d["tables"] or 0) for d in docs)
    figures = sum(int(d["figures"] or 0) for d in docs)
    text = sum(1 for d in docs if d["text_layer"].strip().lower() == "yes")
    print(
        f"{n} documents, {tables} tables, {figures} figures; {text} of {n} with a text layer"
    )

    by_type: dict[str, Counter] = defaultdict(Counter)
    for d in docs:
        by_type[d["type"]]["documents"] += 1
        by_type[d["type"]]["tables"] += int(d["tables"] or 0)
        by_type[d["type"]]["figures"] += int(d["figures"] or 0)
    print(f"\n{'type':<22} {'docs':>5} {'tables':>7} {'figures':>8}")
    for t, c in sorted(by_type.items(), key=lambda kv: -kv[1]["tables"]):
        print(f"{t:<22} {c['documents']:>5} {c['tables']:>7} {c['figures']:>8}")

    status = Counter(d["extracted"].strip().lower() for d in docs)
    print(
        "\nextraction status: "
        + ", ".join(f"{k} {v}" for k, v in sorted(status.items()))
    )

    backlog = [d for d in docs if d["extracted"].strip().lower() != "yes"]
    backlog.sort(key=lambda d: -(int(d["tables"] or 0) + int(d["figures"] or 0)))
    print("\nbacklog (largest first):")
    for d in backlog:
        ocr = "" if d["text_layer"].strip().lower() == "yes" else "  needs OCR"
        print(
            f"  {d['document_id']:<18} {d['year']}  {int(d['tables'] or 0):>4} tables {int(d['figures'] or 0):>3} figures  {d['extracted']}{ocr}"
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
