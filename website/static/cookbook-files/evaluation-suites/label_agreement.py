"""Measure agreement between two labellers and list their disagreements.

Reads a CSV with one row per item and two label columns (label_a,
label_b) and reports the percent agreement, Cohen's kappa (agreement
corrected for chance), and the items the labellers disagree on, which
are the ones to adjudicate and the ones that refine the labelling
guide. Standard library only.

Usage:
    python label_agreement.py labels_two_annotators.csv

What this does not do: two labellers on twenty items give a first
reading; a labelling guide is calibrated on a hundred or more, and a
third labeller adjudicates the disagreements. Kappa depends on the
label distribution, so it is reported with the percent agreement and
the number of items, never alone.
"""

from __future__ import annotations

import argparse
import csv
import sys
from collections import Counter
from pathlib import Path


def kappa(a: list[str], b: list[str]) -> float:
    n = len(a)
    observed = sum(1 for x, y in zip(a, b, strict=True) if x == y) / n
    ca, cb = Counter(a), Counter(b)
    expected = sum(ca[k] * cb[k] for k in set(a) | set(b)) / (n * n)
    return (observed - expected) / (1 - expected) if expected < 1 else 1.0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("labels", type=Path)
    args = parser.parse_args(argv)
    with args.labels.open(newline="", encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh))
    a = [r["label_a"] for r in rows]
    b = [r["label_b"] for r in rows]
    agree = sum(1 for x, y in zip(a, b, strict=True) if x == y)
    k = kappa(a, b)
    print(
        f"{len(rows)} items: agreement {agree / len(rows):.2f} ({agree}/{len(rows)}), Cohen's kappa {k:.2f}"
    )
    disagreements = [r for r in rows if r["label_a"] != r["label_b"]]
    if disagreements:
        print("\ndisagreements to adjudicate:")
        for r in disagreements:
            print(
                f"  {r['item_id']}: {r['text']!r}: A {r['label_a']} / B {r['label_b']}"
            )
    none_split = sum(1 for r in disagreements if "NONE" in (r["label_a"], r["label_b"]))
    if none_split:
        print(
            f"\n{none_split} of {len(disagreements)} disagreements involve NONE: the guide's rule on near concepts needs an example"
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
