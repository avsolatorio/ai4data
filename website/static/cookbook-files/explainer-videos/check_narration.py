"""Check the narration of an explainer video before it is recorded.

Reads a step script (JSON with a list of steps, each narrated step with a
"say" field for the caption and an optional "speak" field for the spoken
form) and the source table of the video (CSV with value and
previous_value columns). It reports:

- the estimated length, from the spoken words at a fixed reading rate plus
  a fixed time per narrated step for the settle and the pause;
- each figure in "say" that is not a value or a previous value of the
  table (reference periods, years, and ages are left out, and thousands
  separators are removed before the comparison);
- digits left in "speak", and a "say" with digits that has no "speak",
  because the voice then decides how to read the figure;
- each sentence of "say" longer than the word limit.

The defaults (3.1 words per second and 1.0 s per step) are the measured
rate of the Kokoro voice af_heart at speed 1.1 in the program's video
tool. Exit status 1 when a figure is not in the table, digits are left
for the voice, a sentence is too long, or the estimate is outside the
target range.

Usage:
    python check_narration.py narration.json release_table.csv
    python check_narration.py narration.json release_table.csv --max-words 20

Requires pandas.

What this does not check: whether a figure is attached to the right series
and period, whether the spoken words say the same figure as the caption,
or whether a provisional value is called provisional. A person reads the
narration against the table and listens to the recording for that.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

import pandas as pd

NUMBER = re.compile(r"(?<![\w-])(\d{1,3}(?:,\d{3})+|\d+(?:\.\d+)?)(?![\w-])")
SKIP = re.compile(
    r"\b(19|20)\d{2}(-Q[1-4])?\b|\baged \d+( to \d+)?\b|\b\d+ and over\b|\bQ[1-4]\b"
)
SENTENCE = re.compile(r"(?<=[.!?])\s+")


def minutes(seconds: float) -> str:
    whole = round(seconds)
    return f"{whole // 60} min {whole % 60:02d} s"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("narration", type=Path)
    parser.add_argument("table", type=Path)
    parser.add_argument("--rate", type=float, default=3.1, help="words per second")
    parser.add_argument(
        "--step-seconds", type=float, default=1.0, help="settle and pause per step"
    )
    parser.add_argument("--min-seconds", type=float, default=180)
    parser.add_argument("--max-seconds", type=float, default=300)
    parser.add_argument("--max-words", type=int, default=25, help="words per sentence")
    args = parser.parse_args(argv)

    steps = json.loads(args.narration.read_text(encoding="utf-8"))["steps"]
    table = pd.read_csv(args.table, dtype=str)
    known = {
        float(v.replace(",", ""))
        for col in ("value", "previous_value")
        for v in table[col].dropna()
    }

    problems: list[str] = []
    words = narrated = figures = found = 0
    for number, step in enumerate(steps, 1):
        say = step.get("say")
        if not say:
            continue
        narrated += 1
        where = f"step {number}, {step.get('js', step.get('do', ''))}"
        speak = step.get("speak")
        words += len((speak or say).split())
        for match in NUMBER.finditer(SKIP.sub(" ", say)):
            figures += 1
            if float(match.group(1).replace(",", "")) in known:
                found += 1
            else:
                problems.append(
                    f"{where}: {match.group(1)} is not a value or a previous value in {args.table.name}"
                )
        if speak and re.search(r"\d", speak):
            digits = ", ".join(re.findall(r"\d[\d.,]*\d|\d", speak))
            problems.append(f"{where}: speak keeps the digits {digits}")
        elif not speak and re.search(r"\d", say):
            problems.append(f"{where}: say has digits and the step has no speak")
        for sentence in SENTENCE.split(say):
            count = len(sentence.split())
            if count > args.max_words:
                problems.append(
                    f"{where}: sentence of {count} words (limit {args.max_words})"
                )

    seconds = words / args.rate + narrated * args.step_seconds
    print(f"{args.narration.name}: {narrated} narrated steps, {words} spoken words")
    print(
        f"estimated length: {minutes(seconds)} at {args.rate} words per second"
        f" and {args.step_seconds} s per step"
        f" (target {minutes(args.min_seconds)} to {minutes(args.max_seconds)})"
    )
    if not args.min_seconds <= seconds <= args.max_seconds:
        problems.append("estimated length is outside the target range")
    print(f"figures in say: {figures}; {found} in the table, {figures - found} not")
    for p in problems:
        print(f"problem: {p}")
    print("result: " + ("FAIL" if problems else "PASS"))
    return 1 if problems else 0


if __name__ == "__main__":
    sys.exit(main())
