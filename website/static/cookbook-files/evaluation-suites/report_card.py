"""Produce a report card for an evaluation run with intervals.

Reads a run (CSV: question_id, language, slice, pass 0|1) and prints,
overall and per language and slice, the pass rate with a 95 percent
bootstrap interval and the number of questions, in a form that can be
pasted into a release note or compared across organizations that use the
same question set format. Standard library only; fixed seed.

Usage:
    python report_card.py run_scores_v2.csv --run-name "search v2, 2026-10-01"

What this does not do: a report card describes one run on one question
set. Comparability across organizations needs the same set, or sets
built with the same rules, and the same measure definitions; the
reporting chapter states what to publish with the numbers.
"""

from __future__ import annotations

import argparse
import csv
import random
import sys
from collections import defaultdict
from pathlib import Path


def interval(vals: list[int], reps: int = 2000, seed: int = 7) -> tuple[float, float]:
    rng = random.Random(seed)
    n = len(vals)
    means = sorted(sum(rng.choice(vals) for _ in range(n)) / n for _ in range(reps))
    return means[int(0.025 * reps)], means[int(0.975 * reps) - 1]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("run", type=Path)
    parser.add_argument("--run-name", default="")
    args = parser.parse_args(argv)
    with args.run.open(newline="", encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh))
    groups: dict[str, list[int]] = defaultdict(list)
    for r in rows:
        groups["all"].append(int(r["pass"]))
        groups[f"language={r['language']}"].append(int(r["pass"]))
        groups[f"slice={r['slice']}"].append(int(r["pass"]))
    title = args.run_name or args.run.name
    print(f"Report card: {title}\n")
    print(f"{'group':<18} {'n':>3} {'pass rate':>9} {'95% interval':>14}")
    for name, vals in groups.items():
        lo, hi = interval(vals)
        print(
            f"{name:<18} {len(vals):>3} {sum(vals) / len(vals):>9.2f} {f'[{lo:.2f}, {hi:.2f}]':>14}"
        )
    print(
        "\nPublish with: the question set version, the measure definitions, the model and index versions, and the date."
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
