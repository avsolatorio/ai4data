"""Check the numbers and claims in a draft commentary against the release table.

Reads the release table (CSV: series, period, value, previous_value,
unit, status) and a draft commentary (Markdown) and reports:

    verified      numbers in the text that match a value or a previous
                  value in the table, or a difference between the two
                  (within rounding), or a rounded form of them
    unverified    numbers in the text that match nothing in the table,
                  which the author must source or remove
    claims        comparative phrases ("lowest since", "highest", "record")
                  that no table value can verify and need a source
    status        whether the text says that provisional values are
                  provisional

Exit status 1 when any number is unverified or a claim is unsourced, so
that a draft cannot be published unchecked. Standard library only.

Usage:
    python commentary_check.py release_table.csv commentary_draft.md

What this does not do: a verified number can be attached to the wrong
series in the text; the analyst reads the draft. Claims about history
("lowest since 2019") need the series history, which the dissemination
cookbook's number check can supply when the full series is given.
"""

from __future__ import annotations

import argparse
import csv
import re
import sys
from pathlib import Path

NUMBER = re.compile(r"(?<![\w.])(\d{1,3}(?:,\d{3})+|\d+)(?:\.(\d+))?(?![\w])")
CLAIM = re.compile(
    r"\b(lowest|highest|record|largest|smallest|fastest|first time)\b[^.]*\.",
    re.IGNORECASE,
)
YEAR = re.compile(r"^(19|20)\d{2}$")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("table", type=Path)
    parser.add_argument("draft", type=Path)
    args = parser.parse_args(argv)
    with args.table.open(newline="", encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh))
    text = args.draft.read_text(encoding="utf-8")

    known: dict[float, str] = {}
    for r in rows:
        v, p = float(r["value"]), float(r["previous_value"])
        known[v] = f"{r['series']} value"
        known[p] = f"{r['series']} previous"
        known[round(v - p, 3)] = f"{r['series']} change"
        if r["unit"].startswith("thousands"):
            known[round(v / 1000, 2)] = f"{r['series']} value in millions"
            known[(v - p) * 1000] = f"{r['series']} change in units"

    verified = unverified = 0
    for whole, frac in NUMBER.findall(text):
        raw = whole + ("." + frac if frac else "")
        if YEAR.match(whole) and not frac:
            continue
        n = float(raw.replace(",", ""))
        match = next(
            (
                label
                for k, label in known.items()
                if abs(k - n) < 1e-9
                or (
                    n >= 1000
                    and abs(k - n) <= 0.5 * 10 ** (len(whole.replace(",", "")) - 2)
                )
            ),
            None,
        )
        if match:
            verified += 1
            print(f"verified   {raw:>8}  {match}")
        else:
            unverified += 1
            print(f"UNVERIFIED {raw:>8}  not in the release table")
    claims = CLAIM.findall(text)
    sentences = [" ".join(m.group(0).split()) for m in CLAIM.finditer(text)]
    for s in sentences:
        print(f"CLAIM      {s[:90]}")
    provisional = any(r["status"] == "P" for r in rows)
    says = "provisional" in text.lower()
    print(
        f"\n{verified} verified, {unverified} unverified, {len(claims)} claim(s) needing a source; provisional values {'are stated as provisional' if says or not provisional else 'are NOT stated as provisional'}"
    )
    return 1 if unverified or claims or (provisional and not says) else 0


if __name__ == "__main__":
    sys.exit(main())
