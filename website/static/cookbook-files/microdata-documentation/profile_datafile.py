"""Draft a data dictionary from a data file.

Reads a CSV data file and writes a draft dictionary with one row per
column: the name, an inferred type (numeric, categorical, or string), the
number of records, the number of empty or coded-missing values, the
distinct codes of a categorical column written as ``code=`` with the label
left for the curator, the candidate missing codes (negative integers), and
the range of a numeric column. Labels, universes, questions, and concepts
are left empty: they come from the questionnaire and the curator, and the
draft shows exactly which ones are needed. Standard library only.

Usage:
    python profile_datafile.py lfs_2025q2_sample.csv draft_dictionary.csv

Exit status 0. The summary on standard output counts the columns by type
and the fields the curator has to fill.

What this does not do: a column with few distinct integer values is
treated as categorical, which is wrong for a small count variable, and a
code list seen in a sample may miss codes that occur only in the full
file. The Metadata Editor imports Stata, SPSS, and other files with their
value labels, which this script cannot read; use it where only a CSV exists.
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

CATEGORICAL_MAX = 12


def is_number(value: str) -> bool:
    try:
        float(value)
    except ValueError:
        return False
    return True


def profile(column: list[str]) -> dict[str, str]:
    present = [v for v in column if v.strip() != ""]
    numeric = all(is_number(v) for v in present) and present
    missing_codes = sorted(
        {v for v in present if numeric and float(v) < 0 and float(v).is_integer()},
        key=float,
    )
    valid = [v for v in present if v not in missing_codes]
    distinct = sorted(set(valid), key=lambda v: (float(v) if numeric else 0, v))
    if (
        numeric
        and len(distinct) <= CATEGORICAL_MAX
        and all(float(v).is_integer() for v in distinct)
    ):
        kind = "categorical"
    elif numeric:
        kind = "numeric"
    elif len(distinct) <= CATEGORICAL_MAX:
        kind = "categorical"
    else:
        kind = "string"
    values = ";".join(f"{v}=" for v in distinct) if kind == "categorical" else ""
    rng = (
        f"{min(map(float, valid)):g} to {max(map(float, valid)):g}"
        if kind == "numeric" and valid
        else ""
    )
    return {
        "type": kind,
        "n": str(len(column)),
        "n_missing": str(len(column) - len(valid)),
        "values": values,
        "missing": ";".join(missing_codes),
        "range": rng,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("datafile", type=Path)
    parser.add_argument("draft", type=Path, help="where to write the draft dictionary")
    args = parser.parse_args(argv)
    with args.datafile.open(newline="", encoding="utf-8") as fh:
        reader = csv.DictReader(fh)
        rows = list(reader)
        names = reader.fieldnames or []
    out_fields = [
        "name",
        "label",
        "type",
        "universe",
        "question",
        "values",
        "missing",
        "concept",
        "n",
        "n_missing",
        "range",
    ]
    drafts = []
    for name in names:
        p = profile([r[name] for r in rows])
        drafts.append(
            {
                "name": name,
                "label": "",
                "universe": "",
                "question": "",
                "concept": "",
                **p,
            }
        )
    with args.draft.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=out_fields)
        writer.writeheader()
        writer.writerows(drafts)
    by_type = {
        k: sum(1 for d in drafts if d["type"] == k)
        for k in ("categorical", "numeric", "string")
    }
    codes = sum(d["values"].count("=") for d in drafts)
    print(
        f"{len(rows)} records, {len(names)} columns: "
        + ", ".join(f"{v} {k}" for k, v in by_type.items())
    )
    print(
        f"to fill by the curator: {len(names)} labels, {len(names)} universes, {codes} value labels; questions and concepts where they apply"
    )
    with_missing = [d["name"] for d in drafts if d["missing"]]
    print(
        f"candidate missing codes found in {len(with_missing)} columns: {', '.join(with_missing)}"
    )
    print(f"draft written to {args.draft}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
