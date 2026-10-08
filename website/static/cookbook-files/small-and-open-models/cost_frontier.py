"""Find the models on the cost-quality frontier for a task.

Reads candidate models (CSV: model, size_b, hosting local|hosted,
suite_score, national_language_score, cost_per_1k_queries,
latency_ms_p50, license) and prints the frontier: the models that no
other model beats on both the suite score and the cost. Then applies a
required score (overall and on the national language) and prints the
cheapest model that meets it, with the saving against the best-scoring
model. Standard library only.

Usage:
    python cost_frontier.py candidates.csv --required 0.90 --required-national 0.85

What this does not do: scores come from the organization's own suite on
its own questions; the numbers here are illustrative. Cost per query is
measured, including local server and staff time, as the running chapter
describes; a frontier drawn from list prices is a sketch.
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("candidates", type=Path)
    parser.add_argument(
        "--required", type=float, default=0.0, help="required suite score"
    )
    parser.add_argument(
        "--required-national",
        type=float,
        default=0.0,
        help="required score on the national language",
    )
    args = parser.parse_args(argv)
    with args.candidates.open(newline="", encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh))
    for r in rows:
        r["score"] = float(r["suite_score"])
        r["nat"] = float(r["national_language_score"])
        r["cost"] = float(r["cost_per_1k_queries"])

    frontier = [
        r
        for r in rows
        if not any(
            o["score"] >= r["score"]
            and o["cost"] <= r["cost"]
            and (o["score"] > r["score"] or o["cost"] < r["cost"])
            for o in rows
        )
    ]
    print(f"{len(rows)} candidates; {len(frontier)} on the cost-quality frontier\n")
    print(
        f"{'model':<14} {'hosting':<7} {'score':>6} {'national':>8} {'cost/1k':>8} {'p50 ms':>7}  {'frontier':<8} licence"
    )
    for r in sorted(rows, key=lambda r: r["cost"]):
        print(
            f"{r['model']:<14} {r['hosting']:<7} {r['score']:>6.2f} {r['nat']:>8.2f} {r['cost']:>8.2f} {r['latency_ms_p50']:>7}  {'yes' if r in frontier else '':<8} {r['license']}"
        )
    eligible = [
        r
        for r in rows
        if r["score"] >= args.required and r["nat"] >= args.required_national
    ]
    if not eligible:
        print(
            f"\nno candidate meets the required scores ({args.required:.2f} overall, {args.required_national:.2f} national); raise the model size or improve the adaptation"
        )
        return 1
    cheapest = min(eligible, key=lambda r: r["cost"])
    best = max(rows, key=lambda r: r["score"])
    print(
        f"\ncheapest candidate meeting {args.required:.2f} overall and {args.required_national:.2f} national: {cheapest['model']} at {cheapest['cost']:.2f} per 1,000 queries"
    )
    print(
        f"best-scoring candidate: {best['model']} at {best['cost']:.2f}; choosing the cheapest passing model saves {(1 - cheapest['cost'] / best['cost']):.0%} per query"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
