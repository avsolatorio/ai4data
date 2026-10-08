"""Merge citation events from identifier services and text mining into one
list of citing documents per dataset.

Reads citation events (CSV: source, dataset_id, dataset_doi, citing_id,
citing_title, citing_year, relation) collected from DataCite, Crossref,
OpenAlex, or any Scholix-conformant service, and from the mention
extraction of this cookbook, and merges them: one row per dataset and
citing document, with the sources that found it. DOIs are compared
case-insensitively. Reports per dataset the unique citing documents, how
many each source found, how many were found by one source only (the ones
to check), and the overlap between the identifier services and text
mining. Standard library only.

Usage:
    python merge_citations.py citation_events.csv

What this does not do: it merges by identifier. Two records of the same
paper with different identifiers (a preprint and the journal version) stay
separate until a mapping says otherwise; the review in the harmonization
chapter handles that. A citation event says a document references the
dataset; what it did with the data is the typology's question.
"""

from __future__ import annotations

import argparse
import csv
import sys
from collections import defaultdict
from pathlib import Path


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("events", type=Path)
    args = parser.parse_args(argv)
    with args.events.open(newline="", encoding="utf-8") as fh:
        events = list(csv.DictReader(fh))

    merged: dict[tuple[str, str], dict] = {}
    for e in events:
        key = (e["dataset_id"], e["citing_id"].strip().lower())
        m = merged.setdefault(
            key,
            {"title": e["citing_title"], "year": e["citing_year"], "sources": set()},
        )
        m["sources"].add(e["source"])

    by_dataset: dict[str, list[dict]] = defaultdict(list)
    for (dataset, _), m in merged.items():
        by_dataset[dataset].append(m)
    sources = sorted({e["source"] for e in events})

    print(
        f"{len(events)} events -> {len(merged)} unique citing documents across {len(by_dataset)} datasets\n"
    )
    print(
        f"{'dataset':<22} {'unique':>6} "
        + " ".join(f"{s:>12}" for s in sources)
        + f" {'one source':>11}"
    )
    single: list[str] = []
    for dataset, items in sorted(by_dataset.items()):
        per = {s: sum(1 for m in items if s in m["sources"]) for s in sources}
        one = [m for m in items if len(m["sources"]) == 1]
        single += [
            f"{dataset}: {m['title']} ({m['year']}) found by {next(iter(m['sources']))} only"
            for m in one
        ]
        print(
            f"{dataset:<22} {len(items):>6} "
            + " ".join(f"{per[s]:>12}" for s in sources)
            + f" {len(one):>11}"
        )
    ids = {
        m_key
        for m_key, m in merged.items()
        if m["sources"] & {"datacite", "crossref", "openalex"}
    }
    mined = {m_key for m_key, m in merged.items() if "text-mining" in m["sources"]}
    print(
        f"\nfound through identifiers {len(ids)}, through text mining {len(mined)}, by both {len(ids & mined)}, by text mining only {len(mined - ids)}"
    )
    if single:
        print(
            "\nfound by one source only (check the record and add the identifier where it is missing):"
        )
        for s in single:
            print(f"  {s}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
