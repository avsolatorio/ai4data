"""Compare a model's quality scores with curators' scores.

Reads two CSV files with the same columns (record_id and one column per
quality dimension, scores 1 to 5): the model's scores and the curators'
scores on the same records. Reports, per dimension, the exact agreement,
the agreement within one point, and the mean difference (model minus
curator). A dimension whose mean difference is 0.5 or more in either
direction is marked as miscalibrated: the model is systematically more or
less severe than the curators. Standard library only.

Usage:
    python assess_agreement.py model_scores.csv curator_scores.csv

What this does not do: five records give a demonstration. A calibration
needs 50 to 100 records scored by two curators, with their own agreement
reported (Cohen's kappa or similar) so that the model is compared with a
reliable reference.
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

THRESHOLD = 0.5


def load(path: Path) -> dict[str, dict[str, float]]:
    with path.open(newline="", encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh))
    return {
        r["record_id"]: {
            k: float(v) for k, v in r.items() if k != "record_id" and v != ""
        }
        for r in rows
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("model", type=Path)
    parser.add_argument("curator", type=Path)
    args = parser.parse_args(argv)
    model = load(args.model)
    curator = load(args.curator)
    records = sorted(set(model) & set(curator))
    if not records:
        sys.exit("no records in common")
    dimensions = [
        d for d in next(iter(model.values())) if all(d in curator[r] for r in records)
    ]

    print(f"{len(records)} records, {len(dimensions)} dimensions\n")
    print(f"{'dimension':<22} {'exact':>6} {'within 1':>9} {'mean diff':>10}  note")
    flagged = 0
    for d in dimensions:
        pairs = [(model[r][d], curator[r][d]) for r in records]
        exact = sum(1 for m, c in pairs if m == c) / len(pairs)
        within = sum(1 for m, c in pairs if abs(m - c) <= 1) / len(pairs)
        diff = sum(m - c for m, c in pairs) / len(pairs)
        note = ""
        if abs(diff) >= THRESHOLD:
            flagged += 1
            note = "model more severe" if diff < 0 else "model more lenient"
        print(f"{d:<22} {exact:>6.2f} {within:>9.2f} {diff:>+10.2f}  {note}")
    print(
        f"\n{flagged} dimension(s) miscalibrated at a mean difference of {THRESHOLD} or more"
    )
    print(
        "Scores are compared with the curators' scores as the reference; the curators' own agreement is reported separately."
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
