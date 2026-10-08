"""Check a data dictionary before it is published.

Reads a data dictionary with one row per variable (CSV) and reports what an
AI system or a user would find missing or misleading:

errors (exit status 1)
  - a variable without a label
  - a label that repeats the variable name
  - duplicate variable names within a file
  - a categorical variable without value labels

warnings
  - a numeric variable without a statement of its missing-value codes
  - a variable without a universe
  - a collected variable without its question text
  - a variable without a concept, where a concept map could link it to a
    classification

Columns expected: file_id, name, label, type (numeric | categorical |
string), universe, question, values ("code=label;code=label"), missing
("code;code"), concept. Other columns are ignored. Standard library only.

What this does not check: whether the labels are correct, whether the
value labels match the data, or whether the universe statements agree with
the skip patterns. Those need the data file and the questionnaire.

Usage:
    python check_dictionary.py lfs_2025q2_dictionary.csv
"""

from __future__ import annotations

import argparse
import csv
import sys
from collections import Counter
from pathlib import Path

ID_LIKE = ("id",)


def is_identifier(name: str) -> bool:
    return name.lower().endswith(ID_LIKE)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("dictionary", type=Path, help="CSV with one row per variable")
    args = parser.parse_args(argv)

    with args.dictionary.open(newline="", encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh))
    if not rows:
        sys.exit("no variables found")

    errors: list[str] = []
    warnings: list[str] = []
    names = Counter((r["file_id"], r["name"]) for r in rows)
    for (file_id, name), n in names.items():
        if n > 1:
            errors.append(f"{file_id}/{name}: name appears {n} times")

    for r in rows:
        name = r["name"].strip()
        label = (r.get("label") or "").strip()
        vtype = (r.get("type") or "").strip().lower()
        ref = f"{r['file_id']}/{name}"
        if not label:
            errors.append(f"{ref}: no label")
        elif label.lower() == name.lower():
            errors.append(f"{ref}: label repeats the name ({label!r})")
        if vtype == "categorical" and not (r.get("values") or "").strip():
            errors.append(f"{ref}: categorical variable without value labels")
        if (
            vtype == "numeric"
            and not (r.get("missing") or "").strip()
            and not is_identifier(name)
        ):
            warnings.append(
                f"{ref}: numeric variable without a missing-value statement"
            )
        if not (r.get("universe") or "").strip():
            warnings.append(f"{ref}: no universe")
        if not (r.get("question") or "").strip() and not is_identifier(name):
            warnings.append(f"{ref}: no question or derivation text")
        if (
            vtype == "categorical"
            and not (r.get("concept") or "").strip()
            and not is_identifier(name)
        ):
            warnings.append(f"{ref}: no concept or classification named")

    print(f"{len(rows)} variables in {len({r['file_id'] for r in rows})} file(s)")
    filled = {
        col: sum(1 for r in rows if (r.get(col) or "").strip())
        for col in ("label", "universe", "question", "values", "missing", "concept")
    }
    for col, n in filled.items():
        print(f"  {col:<10} {n:>4} of {len(rows)}  {n / len(rows):>4.0%}")
    print()
    for w in warnings:
        print(f"warning  {w}")
    for e in errors:
        print(f"error    {e}")
    print(f"\n{len(errors)} error(s), {len(warnings)} warning(s)")
    print(
        "Not checked here: whether labels and value labels are correct, or whether universes match the skip patterns."
    )
    return 1 if errors else 0


if __name__ == "__main__":
    sys.exit(main())
