"""Check the numbers in a generated answer against retrieved values.

Takes an answer text and a CSV of the values that were retrieved for it,
with SDMX cross-domain concept names as columns (SERIES, REF_AREA,
TIME_PERIOD, OBS_VALUE, UNIT_MEASURE). Every number in the answer is matched
to a retrieved value within a rounding tolerance and marked verified or
unverified. Standard library only.

This is the simplest form of the check described in the Proof-Carrying
Numbers paper (https://arxiv.org/abs/2509.06902). Production systems also
check that the series and period named in the sentence match the value.

Usage:
    python verify_numbers.py answer.txt example_values.csv
    echo "Undernourishment fell from 8.4% in 2022 to 7.2% in 2024." \\
        | python verify_numbers.py - example_values.csv
"""

import argparse
import csv
import re
import sys

NUMBER = re.compile(r"-?\d[\d,]*(?:\.\d+)?")
YEAR = re.compile(r"^(19|20)\d\d$")


def find_numbers(text):
    """Yield number strings, skipping unit text such as "per 1,000"."""
    for m in NUMBER.finditer(text):
        if text[max(0, m.start() - 4) : m.start()].lower() == "per ":
            continue
        yield m.group().rstrip(",")


def parse(raw):
    return float(raw.replace(",", ""))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("answer", help="text file, or - for stdin")
    parser.add_argument("values", help="CSV of retrieved values")
    parser.add_argument("--tolerance", type=float, default=0.05)
    args = parser.parse_args()

    text = sys.stdin.read() if args.answer == "-" else open(args.answer, encoding="utf-8").read()
    with open(args.values, newline="", encoding="utf-8") as fh:
        values = list(csv.DictReader(fh))

    found = list(find_numbers(text))
    print(f"{'number':>8}  {'status':<10} match")
    unverified = 0
    for raw in found:
        if YEAR.match(raw):
            print(f"{raw:>8}  {'period':<10}")
            continue
        num = parse(raw)
        match = None
        for v in values:
            if abs(float(v["OBS_VALUE"]) - num) <= args.tolerance:
                match = v
                break
        if match:
            print(
                f"{raw:>8}  {'verified':<10} {match['SERIES']} "
                f"{match['REF_AREA']} {match['TIME_PERIOD']} = {match['OBS_VALUE']} {match['UNIT_MEASURE']}"
            )
        else:
            unverified += 1
            print(f"{raw:>8}  {'UNVERIFIED':<10} no retrieved value within {args.tolerance}")

    print(f"\n{len(found)} numbers, {unverified} unverified")
    sys.exit(1 if unverified else 0)


if __name__ == "__main__":
    main()
