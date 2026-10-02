"""Check the numbers in a generated answer against retrieved values.

Takes an answer text and a CSV of the values that were retrieved for it,
with SDMX cross-domain concept names as columns (SERIES, REF_AREA,
TIME_PERIOD, OBS_VALUE, UNIT_MEASURE). Every number in the answer is matched
to a retrieved value within a tolerance and marked verified or unverified.
Standard library only.

The tolerance is the larger of an absolute and a relative allowance, so that
7.6 reported as 7.60 passes and 12,500 reported as 12,480 passes at 0.2
percent, while 7.6 reported as 7.2 fails.

This is the simplest form of the check described in the Proof-Carrying
Numbers paper (https://arxiv.org/abs/2509.06902). Production systems also
check that the series and period named in the sentence match the value.

Usage:
    python verify_numbers.py answer.txt example_values.csv
    echo "Undernourishment fell from 8.4% in 2022 to 7.2% in 2024." \\
        | python verify_numbers.py - example_values.csv

Exit status is 0 when every number is verified, 1 otherwise.
"""

from __future__ import annotations

import argparse
import csv
import re
import sys
from collections.abc import Iterator
from pathlib import Path

NUMBER = re.compile(r"-?\d[\d,]*(?:\.\d+)?")
YEAR = re.compile(r"^(19|20)\d\d$")


def find_numbers(text: str) -> Iterator[str]:
    """Yield number strings, skipping unit text such as "per 1,000"."""
    for match in NUMBER.finditer(text):
        if text[max(0, match.start() - 4) : match.start()].lower() == "per ":
            continue
        yield match.group().rstrip(",")


def parse(raw: str) -> float:
    return float(raw.replace(",", ""))


def matches(number: float, value: float, abs_tol: float, rel_tol: float) -> bool:
    return abs(value - number) <= max(abs_tol, rel_tol * abs(value))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("answer", help="text file, or - for stdin")
    parser.add_argument("values", type=Path, help="CSV of retrieved values")
    parser.add_argument("--abs-tolerance", type=float, default=0.05, help="absolute allowance (default 0.05)")
    parser.add_argument(
        "--rel-tolerance", type=float, default=0.002, help="relative allowance (default 0.2%%)"
    )
    args = parser.parse_args(argv)

    text = sys.stdin.read() if args.answer == "-" else Path(args.answer).read_text(encoding="utf-8")
    with args.values.open(newline="", encoding="utf-8") as fh:
        values = list(csv.DictReader(fh))

    unverified = 0
    found = list(find_numbers(text))
    print(f"{'number':>8}  {'status':<10} match")
    for raw in found:
        if YEAR.match(raw):
            print(f"{raw:>8}  {'period':<10}")
            continue
        number = parse(raw)
        match = next(
            (
                v
                for v in values
                if matches(number, float(v["OBS_VALUE"]), args.abs_tolerance, args.rel_tolerance)
            ),
            None,
        )
        if match:
            print(
                f"{raw:>8}  {'verified':<10} {match['SERIES']} {match['REF_AREA']} "
                f"{match['TIME_PERIOD']} = {match['OBS_VALUE']} {match['UNIT_MEASURE']}"
            )
        else:
            unverified += 1
            print(f"{raw:>8}  {'UNVERIFIED':<10} no retrieved value within tolerance")

    print(f"\n{len(found)} numbers, {unverified} unverified")
    return 1 if unverified else 0


if __name__ == "__main__":
    sys.exit(main())
