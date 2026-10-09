"""Assign records to training, validation, and test splits by group.

Reads a CSV, assigns each record to a split by hashing a group key (for
example the household identifier), so that all records of a group fall in
one split and the assignment is the same on every run and every machine.
Writes a file with one row per record (identifier and split), and prints
the share of records per split, a check that no group appears in two
splits, and the distribution of a stratifying variable in each split.
Standard library only.

Usage:
    python make_splits.py data.csv --id record_id --group hhid \
        --stratify region --test 0.2 --validation 0.1 -o splits.csv

With --time-column and --test-from, records whose value in that column is
at or after the given value go to the test split and the rest are split
by group between training and validation.

What this does not do: it does not balance the splits by label or by
stratifier (a grouped split cannot, exactly), it does not check for
leakage through other columns (a derived variable that encodes the
label), and it does not decide the split sizes. The chapter explains the
choices.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import sys
from collections import Counter, defaultdict
from pathlib import Path


def bucket(key: str, seed: str) -> float:
    """Return a stable number in [0, 1) for a key."""
    digest = hashlib.sha256(f"{seed}:{key}".encode()).digest()
    return int.from_bytes(digest[:8], "big") / 2**64


def assign(
    rows: list[dict],
    id_col: str,
    group_col: str,
    test: float,
    validation: float,
    seed: str,
    time_col: str | None,
    test_from: str | None,
) -> dict[str, str]:
    """Return a mapping from record identifier to split name."""
    splits: dict[str, str] = {}
    for r in rows:
        if time_col and test_from and r[time_col] >= test_from:
            splits[r[id_col]] = "test"
            continue
        u = bucket(r[group_col], seed)
        if time_col and test_from:
            splits[r[id_col]] = "validation" if u < validation else "train"
        elif u < test:
            splits[r[id_col]] = "test"
        elif u < test + validation:
            splits[r[id_col]] = "validation"
        else:
            splits[r[id_col]] = "train"
    return splits


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("data")
    parser.add_argument("--id", required=True, help="column that identifies a record")
    parser.add_argument(
        "--group",
        required=True,
        help="column whose values must not be split (household, firm, document)",
    )
    parser.add_argument(
        "--stratify", help="column whose distribution is reported per split"
    )
    parser.add_argument("--test", type=float, default=0.2)
    parser.add_argument("--validation", type=float, default=0.1)
    parser.add_argument(
        "--seed",
        default="v1",
        help="any string; change it to draw a different assignment",
    )
    parser.add_argument("--time-column", help="column for a time-based test split")
    parser.add_argument(
        "--test-from", help="records with time-column >= this value form the test split"
    )
    parser.add_argument("-o", "--output", default="splits.csv")
    args = parser.parse_args(argv)

    with open(args.data, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    splits = assign(
        rows,
        args.id,
        args.group,
        args.test,
        args.validation,
        args.seed,
        args.time_column,
        args.test_from,
    )

    counts = Counter(splits.values())
    n = len(rows)
    print(
        f"{n} records, {len({r[args.group] for r in rows})} groups by '{args.group}', seed '{args.seed}'"
    )
    for name in ("train", "validation", "test"):
        print(f"  {name:<11} {counts.get(name, 0):>5}  {counts.get(name, 0) / n:5.2f}")

    by_group: dict[str, set[str]] = defaultdict(set)
    for r in rows:
        by_group[r[args.group]].add(splits[r[args.id]])
    leaks = [g for g, s in by_group.items() if len(s) > 1]
    print(
        f"groups in more than one split: {len(leaks)}" + (" (leakage)" if leaks else "")
    )

    if args.stratify:
        print(f"distribution of '{args.stratify}' per split (share of the split):")
        cats = sorted({r[args.stratify] for r in rows})
        per = {
            name: Counter(r[args.stratify] for r in rows if splits[r[args.id]] == name)
            for name in counts
        }
        print(
            "  "
            + f"{'category':<12}"
            + "".join(f"{name:>12}" for name in ("train", "validation", "test"))
        )
        for c in cats:
            line = f"  {c:<12}"
            for name in ("train", "validation", "test"):
                total = counts.get(name, 0)
                line += (
                    f"{(per.get(name, Counter())[c] / total if total else 0):>12.2f}"
                )
            print(line)

    with open(args.output, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow([args.id, "split"])
        for r in rows:
            w.writerow([r[args.id], splits[r[args.id]]])
    print(f"splits written to {Path(args.output).name}")
    return 1 if leaks else 0


if __name__ == "__main__":
    sys.exit(main())
