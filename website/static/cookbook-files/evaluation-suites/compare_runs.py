"""Compare two evaluation runs and decide whether a change may ship.

Reads two runs of the same question set (CSV: question_id, language,
slice, pass 0|1) and reports, overall and per language and slice, the
pass rate of each run, the difference, and a bootstrap confidence
interval on the difference over the paired questions. The gate rule:
the change ships when no slice's interval lies entirely below zero
minus the tolerated regression (exit 1 otherwise). A slice that dropped
beyond the tolerance on too few questions for the interval to decide
sends the change to a person (exit 2). Standard library only; the
bootstrap uses a fixed seed so that the result is reproducible.

Usage:
    python compare_runs.py run_scores_v1.csv run_scores_v2.csv --tolerance 0.05

What this does not do: thirty questions give wide intervals, which is
the point: the script shows how little thirty questions can decide. A
slice that the set does not cover (a language with two questions) is
not protected by the gate, and the question chapter is where the set
grows.
"""

from __future__ import annotations

import argparse
import csv
import random
import sys
from collections import defaultdict
from pathlib import Path


def load(path: Path) -> dict[str, dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as fh:
        return {r["question_id"]: r for r in csv.DictReader(fh)}


def interval(diffs: list[int], reps: int = 2000, seed: int = 7) -> tuple[float, float]:
    rng = random.Random(seed)
    n = len(diffs)
    means = sorted(sum(rng.choice(diffs) for _ in range(n)) / n for _ in range(reps))
    return means[int(0.025 * reps)], means[int(0.975 * reps) - 1]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("before", type=Path)
    parser.add_argument("after", type=Path)
    parser.add_argument(
        "--tolerance",
        type=float,
        default=0.05,
        help="regression tolerated on any slice",
    )
    args = parser.parse_args(argv)
    a, b = load(args.before), load(args.after)
    ids = sorted(set(a) & set(b))
    groups: dict[str, list[str]] = defaultdict(list)
    for q in ids:
        groups["all"].append(q)
        groups[f"language={a[q]['language']}"].append(q)
        groups[f"slice={a[q]['slice']}"].append(q)

    print(f"{len(ids)} paired questions; tolerance {args.tolerance:.2f}\n")
    print(
        f"{'group':<18} {'n':>3} {'before':>7} {'after':>6} {'diff':>6} {'95% interval':>16}  verdict"
    )
    blocking: list[str] = []
    review: list[str] = []
    for name, qs in groups.items():
        before = sum(int(a[q]["pass"]) for q in qs) / len(qs)
        after = sum(int(b[q]["pass"]) for q in qs) / len(qs)
        diffs = [int(b[q]["pass"]) - int(a[q]["pass"]) for q in qs]
        lo, hi = interval(diffs)
        verdict = ""
        if hi < -args.tolerance:
            verdict = "REGRESSION"
            blocking.append(name)
        elif after - before < -args.tolerance:
            verdict = "possible regression, too few to decide"
            review.append(name)
        elif len(qs) < 30:
            verdict = "ok (small)"
        else:
            verdict = "ok"
        print(
            f"{name:<18} {len(qs):>3} {before:>7.2f} {after:>6.2f} {after - before:>+6.2f} {f'[{lo:+.2f}, {hi:+.2f}]':>16}  {verdict}"
        )
    print()
    if blocking:
        print(f"gate: BLOCKED by {', '.join(blocking)}")
        return 1
    if review:
        print(
            f"gate: REVIEW ({', '.join(review)} dropped beyond the tolerance on too few questions to decide; a person decides, and the set grows there)"
        )
        return 2
    print(
        "gate: PASS (no slice shows a regression beyond the tolerance; small slices show direction only)"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
