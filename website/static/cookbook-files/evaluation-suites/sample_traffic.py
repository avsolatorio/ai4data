"""Draw the monthly review sample from logged traffic, stratified.

Reads a traffic log (CSV: query_id, timestamp, language, intent,
answered yes|no, user_rating up|down|empty) and draws a review sample
with a fixed seed: every language and intent is represented in
proportion to traffic with a minimum per stratum, and every query the
system declined or a user rated down is included, because those are
where the errors are. Writes the sample and prints what it contains.
Standard library only.

Usage:
    python sample_traffic.py traffic_log.csv --size 12 --min-per-stratum 1 -o review_sample.csv

What this does not do: a sample reviewed by people finds the errors in
the sample. The rates it yields apply to the month's traffic with the
uncertainty of its size, and the oversampled declined and down-rated
queries are reported separately so that they do not inflate the error
rate of ordinary traffic.
"""

from __future__ import annotations

import argparse
import csv
import random
import sys
from collections import defaultdict
from pathlib import Path


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("log", type=Path)
    parser.add_argument("--size", type=int, default=30)
    parser.add_argument("--min-per-stratum", type=int, default=2)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("-o", "--output", type=Path)
    args = parser.parse_args(argv)
    with args.log.open(newline="", encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh))
    rng = random.Random(args.seed)

    flagged = [r for r in rows if r["answered"] == "no" or r["user_rating"] == "down"]
    chosen = {r["query_id"]: dict(r, reason="declined or rated down") for r in flagged}
    strata: dict[tuple[str, str], list[dict[str, str]]] = defaultdict(list)
    for r in rows:
        strata[(r["language"], r["intent"])].append(r)
    remaining = max(args.size - len(chosen), 0)
    for key, group in sorted(strata.items()):
        pool = [r for r in group if r["query_id"] not in chosen]
        share = round(remaining * len(group) / len(rows))
        take = min(len(pool), max(args.min_per_stratum, share))
        for r in rng.sample(pool, take):
            chosen[r["query_id"]] = dict(r, reason=f"stratum {key[0]}/{key[1]}")
    sample = sorted(chosen.values(), key=lambda r: r["query_id"])
    if args.output:
        with args.output.open("w", newline="", encoding="utf-8") as fh:
            w = csv.DictWriter(fh, fieldnames=list(sample[0]))
            w.writeheader()
            w.writerows(sample)
    print(
        f"{len(rows)} logged queries -> sample of {len(sample)}: {len(flagged)} declined or rated down (always included), {len(sample) - len(flagged)} drawn by stratum"
    )
    by = defaultdict(int)
    for r in sample:
        by[(r["language"], r["intent"])] += 1
    print(
        "by stratum: " + ", ".join(f"{k[0]}/{k[1]} {v}" for k, v in sorted(by.items()))
    )
    print(
        "report the error rate of the stratified part and of the flagged part separately"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
