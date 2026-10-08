"""Score retrieval runs by slice: language, paraphrase, and run.

Reads retrieval results (CSV: run, query_id, language, paraphrase yes|no,
expected_id, ranked_ids separated by ";") and reports Recall@1, Recall@3,
and MRR for every run overall and per slice, so that a change that
helps English paraphrases and hurts French official titles is visible.
Standard library only.

Usage:
    python retrieval_by_slice.py retrieval_runs.csv

What this does not do: ten queries per run demonstrate the slices; a
slice needs thirty or more queries before a difference between runs
means anything, and the comparison script of the gates chapter puts an
interval around the difference. Queries with no correct answer (NONE)
are scored by the refusal measures, not here.
"""

from __future__ import annotations

import argparse
import csv
import sys
from collections import defaultdict
from pathlib import Path


def scores(rows: list[dict[str, str]]) -> tuple[float, float, float]:
    r1 = r3 = mrr = 0.0
    for r in rows:
        ranked = [x for x in r["ranked_ids"].split(";") if x]
        if r["expected_id"] in ranked:
            rank = ranked.index(r["expected_id"]) + 1
            r1 += rank == 1
            r3 += rank <= 3
            mrr += 1 / rank
    n = len(rows) or 1
    return r1 / n, r3 / n, mrr / n


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("results", type=Path)
    args = parser.parse_args(argv)
    with args.results.open(newline="", encoding="utf-8") as fh:
        rows = [r for r in csv.DictReader(fh) if r["expected_id"] != "NONE"]
    runs = sorted({r["run"] for r in rows})
    slices: dict[str, dict[str, list]] = defaultdict(lambda: defaultdict(list))
    for r in rows:
        slices["all"][r["run"]].append(r)
        slices[f"language={r['language']}"][r["run"]].append(r)
        slices[f"paraphrase={r['paraphrase']}"][r["run"]].append(r)
    print(f"{len(rows)} scored queries, runs: {', '.join(runs)}\n")
    print(f"{'slice':<16} {'run':<4} {'n':>3} {'R@1':>6} {'R@3':>6} {'MRR':>6}")
    for name, by_run in slices.items():
        for run in runs:
            rs = by_run.get(run, [])
            if not rs:
                continue
            r1, r3, mrr = scores(rs)
            print(f"{name:<16} {run:<4} {len(rs):>3} {r1:>6.2f} {r3:>6.2f} {mrr:>6.2f}")
    print(
        "\nA slice with fewer than thirty queries shows direction; the gate script puts an interval around the difference."
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
