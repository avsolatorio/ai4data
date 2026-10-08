"""Score an extraction run against gold spans, with an error taxonomy.

Reads aligned gold and predicted spans (CSV: doc_id, gold_span,
gold_type, pred_span, pred_type; a row with an empty gold is a spurious
prediction, a row with an empty prediction is a miss) and reports
precision, recall, and F1 under two rules (exact span and type;
overlapping span and type), and the counts of each error kind:

    missed      a gold span with no prediction
    spurious    a prediction with no gold span
    boundary    overlapping spans, same type (a partial match)
    wrong type  same or overlapping span, different type

The taxonomy says what to fix: boundary errors point to the model's span
rules, wrong types to the label definitions, spurious predictions to the
threshold, misses to coverage. Standard library only.

Usage:
    python extraction_scores.py extraction_gold_pred.csv

What this does not do: the alignment between gold and predictions is
given in the file; producing it from two separate lists is the matching
step of an evaluation harness. Twelve rows demonstrate the taxonomy; a
measure needs a labelled sample of a few hundred spans across document
types and languages.
"""

from __future__ import annotations

import argparse
import csv
import sys
from collections import Counter
from pathlib import Path


def overlap(a: str, b: str) -> bool:
    a, b = a.lower().strip(), b.lower().strip()
    return bool(a and b) and (
        a in b or b in a or len(set(a.split()) & set(b.split())) >= 2
    )


def prf(tp: int, fp: int, fn: int) -> tuple[float, float, float]:
    p = tp / (tp + fp) if tp + fp else 0.0
    r = tp / (tp + fn) if tp + fn else 0.0
    f = 2 * p * r / (p + r) if p + r else 0.0
    return p, r, f


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("aligned", type=Path)
    args = parser.parse_args(argv)
    with args.aligned.open(newline="", encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh))

    kinds = Counter()
    exact = Counter()
    lenient = Counter()
    for r in rows:
        g, p = r["gold_span"].strip(), r["pred_span"].strip()
        if g and not p:
            kinds["missed"] += 1
            exact["fn"] += 1
            lenient["fn"] += 1
        elif p and not g:
            kinds["spurious"] += 1
            exact["fp"] += 1
            lenient["fp"] += 1
        elif r["gold_type"] != r["pred_type"]:
            kinds["wrong type"] += 1
            exact["fp"] += 1
            exact["fn"] += 1
            lenient["fp"] += 1
            lenient["fn"] += 1
        elif g.lower() == p.lower():
            kinds["correct"] += 1
            exact["tp"] += 1
            lenient["tp"] += 1
        elif overlap(g, p):
            kinds["boundary"] += 1
            exact["fp"] += 1
            exact["fn"] += 1
            lenient["tp"] += 1
        else:
            kinds["wrong span"] += 1
            exact["fp"] += 1
            exact["fn"] += 1
            lenient["fp"] += 1
            lenient["fn"] += 1

    print(f"{len(rows)} aligned rows\n")
    for name, c in (
        ("exact span and type", exact),
        ("overlapping span, same type", lenient),
    ):
        p, r, f = prf(c["tp"], c["fp"], c["fn"])
        print(
            f"{name:<30} precision {p:.2f}  recall {r:.2f}  F1 {f:.2f}  (tp {c['tp']}, fp {c['fp']}, fn {c['fn']})"
        )
    print("\nerror taxonomy: " + ", ".join(f"{k} {v}" for k, v in kinds.most_common()))
    print(
        "boundary -> span rules; wrong type -> label definitions; spurious -> threshold; missed -> coverage"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
