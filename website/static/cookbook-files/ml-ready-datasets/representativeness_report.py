"""Compare the composition of a dataset with population benchmarks.

Reads a dataset (CSV) and a benchmarks file with one row per variable and
category (variable, category, population_share, source). For each
variable it prints the dataset's share of each category, unweighted and
weighted where a weight column is given, next to the population share
and the difference in percentage points. It flags categories whose share
differs from the population by more than a threshold and categories with
fewer records than a minimum. Exit status 1 when any category is below
the minimum count. Standard library only.

Usage:
    python representativeness_report.py data.csv census_benchmarks.csv \
        --weight weight --min-count 30 --max-diff 5

What this does not do: it compares marginal shares, one variable at a
time. Joint coverage (women aged 55 to 64 in the Northern Region) needs
cross-tabulations the chapter describes, and representativeness of the
label (which occupations appear) is reported separately. A benchmark is
only as current as its source.
"""

from __future__ import annotations

import argparse
import csv
import sys
from collections import defaultdict


def shares(
    rows: list[dict], var: str, weight: str | None
) -> tuple[dict[str, int], dict[str, float]]:
    """Return counts and weighted shares per category of a variable."""
    counts: dict[str, int] = defaultdict(int)
    wsum: dict[str, float] = defaultdict(float)
    for r in rows:
        counts[r[var]] += 1
        wsum[r[var]] += float(r[weight]) if weight else 1.0
    total = sum(wsum.values())
    return dict(counts), {c: w / total for c, w in wsum.items()}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("data")
    parser.add_argument("benchmarks")
    parser.add_argument("--weight", help="survey weight column")
    parser.add_argument(
        "--min-count",
        type=int,
        default=30,
        help="categories with fewer records are flagged",
    )
    parser.add_argument(
        "--max-diff",
        type=float,
        default=5.0,
        help="difference in percentage points that is flagged",
    )
    args = parser.parse_args(argv)

    with open(args.data, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    bench: dict[str, dict[str, float]] = defaultdict(dict)
    sources: dict[str, str] = {}
    with open(args.benchmarks, newline="", encoding="utf-8") as f:
        for b in csv.DictReader(f):
            bench[b["variable"]][b["category"]] = float(b["population_share"])
            sources[b["variable"]] = b.get("source", "")

    n = len(rows)
    print(
        f"{n} records; benchmarks for {len(bench)} variable(s)"
        + (f"; weighted by '{args.weight}'" if args.weight else "")
    )
    small: list[str] = []
    off: list[str] = []
    for var, pop in bench.items():
        if var not in rows[0]:
            print(f"\n{var}: not in the dataset")
            continue
        counts, _ = shares(rows, var, None)
        _, wshare = shares(rows, var, args.weight) if args.weight else (None, None)
        print(f"\n{var}  (benchmark: {sources.get(var, '')})")
        head = (
            f"  {'category':<10}{'n':>6}{'dataset':>10}"
            + (f"{'weighted':>10}" if args.weight else "")
            + f"{'population':>12}{'diff pp':>9}"
        )
        print(head)
        for cat in sorted(set(pop) | set(counts)):
            c = counts.get(cat, 0)
            d = c / n
            p = pop.get(cat)
            diff = (d - p) * 100 if p is not None else None
            line = f"  {cat:<10}{c:>6}{d:>10.3f}"
            if args.weight:
                line += f"{wshare.get(cat, 0.0):>10.3f}"
            line += f"{(f'{p:.3f}' if p is not None else '-'):>12}{(f'{diff:+.1f}' if diff is not None else '-'):>9}"
            flags = []
            if c < args.min_count:
                flags.append(f"fewer than {args.min_count} records")
                small.append(f"{var}={cat} ({c})")
            if diff is not None and abs(diff) > args.max_diff:
                flags.append("differs from the population")
                off.append(f"{var}={cat} ({diff:+.1f} pp)")
            print(line + ("  " + "; ".join(flags) if flags else ""))

    print()
    print(
        f"categories below {args.min_count} records: {len(small)}"
        + (": " + ", ".join(small) if small else "")
    )
    print(
        f"categories more than {args.max_diff:g} pp from the population (unweighted): {len(off)}"
        + (": " + ", ".join(off) if off else "")
    )
    print(
        "A dataset is not a sample of the population; the card states these differences and the weights, and the user decides."
    )
    return 1 if small else 0


if __name__ == "__main__":
    sys.exit(main())
