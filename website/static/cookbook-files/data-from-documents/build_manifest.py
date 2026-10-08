"""Build the work manifest for a document-extraction pipeline run.

Reads the document inventory (document_id, title, year, type, pages,
text_layer, tables, figures, extracted, data_sources) and the processing
status log (document_id, stage, status, updated), and writes the manifest
of what the next run does: for each document the next stage that is not
done, in priority order. Stages run in a fixed order: text, layout,
extract, verify, publish. A failed stage is retried and marked as such.
Documents already published are skipped, which makes a run resumable.

Priority is by use first (number of data sources the document feeds),
then by what is missing (a document with no text layer needs OCR, which
is the slow step and is scheduled first so that it runs while the rest is
reviewed), then by size (fewer pages first, so that results arrive early).
Prints the manifest and the totals: pages to OCR, pages to lay out,
tables to extract. Standard library only.

Usage:
    python build_manifest.py document_inventory.csv processing_status.csv -o manifest.csv

What this does not do: it orders work and counts pages; it does not run
anything. Cost per page comes from the organization's own measurements of
each stage (recipe 9.2), and the priority rule is the one the
organization writes, which this script makes explicit and repeatable.
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

STAGES = ["text", "layout", "extract", "verify", "publish"]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("inventory", type=Path)
    parser.add_argument("status", type=Path)
    parser.add_argument("-o", "--output", type=Path)
    args = parser.parse_args(argv)
    with args.inventory.open(newline="", encoding="utf-8") as fh:
        docs = list(csv.DictReader(fh))
    done: dict[str, dict[str, str]] = {}
    with args.status.open(newline="", encoding="utf-8") as fh:
        for r in csv.DictReader(fh):
            done.setdefault(r["document_id"], {})[r["stage"]] = r["status"]

    manifest: list[dict[str, str]] = []
    for d in docs:
        states = done.get(d["document_id"], {})
        if states.get("publish") == "done":
            continue
        next_stage = next(s for s in STAGES if states.get(s) != "done")
        retry = states.get(next_stage) == "failed"
        needs_ocr = next_stage == "text" and d["text_layer"].strip().lower() == "no"
        sources = [s for s in d["data_sources"].split(";") if s.strip()]
        manifest.append(
            {
                "document_id": d["document_id"],
                "next_stage": next_stage,
                "retry": "yes" if retry else "",
                "ocr": "yes" if needs_ocr else "",
                "pages": d["pages"],
                "tables": d["tables"],
                "data_sources": len(sources),
                "reason": ("retry after failure; " if retry else "")
                + ("OCR needed; " if needs_ocr else "")
                + f"feeds {len(sources)} data source(s)",
            }
        )
    manifest.sort(
        key=lambda m: (-int(m["data_sources"]), 0 if m["ocr"] else 1, int(m["pages"]))
    )
    fields = list(manifest[0]) if manifest else []
    out = (
        args.output.open("w", newline="", encoding="utf-8")
        if args.output
        else sys.stdout
    )
    writer = csv.DictWriter(out, fieldnames=fields)
    writer.writeheader()
    writer.writerows(manifest)
    if args.output:
        out.close()
    ocr_pages = sum(int(m["pages"]) for m in manifest if m["ocr"])
    layout_pages = sum(
        int(m["pages"]) for m in manifest if STAGES.index(m["next_stage"]) <= 1
    )
    tables = sum(
        int(m["tables"]) for m in manifest if STAGES.index(m["next_stage"]) <= 2
    )
    skipped = len(docs) - len(manifest)
    print(
        f"{len(docs)} documents in the inventory, {skipped} published and skipped, {len(manifest)} in the manifest",
        file=sys.stderr,
    )
    print(
        f"pages to OCR {ocr_pages}, pages to lay out {layout_pages}, tables to extract {tables}, retries {sum(1 for m in manifest if m['retry'])}",
        file=sys.stderr,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
