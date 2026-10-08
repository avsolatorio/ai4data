"""Validate an automatic grader against human scores and look for bias.

Reads a CSV with one row per graded item (item_id, human_score 0|1,
judge_score 0|1, answer_length in words, position A|B for the order in
which the answer was shown in a pairwise or listed layout) and reports
agreement and Cohen's kappa between the grader and the humans, the
grader's false-pass and false-fail rates, and two bias checks: the pass
rate of the grader by answer length (short versus long) and by position,
compared with the humans' pass rates in the same groups. Standard library
only.

Usage:
    python judge_agreement.py judge_vs_human.csv

What this does not do: twenty items show the shape of the checks; a
grader is validated on a hundred or more human-scored items per task
and re-validated when the model or the rubric changes. Agreement with
humans bounds the grader at the humans' own consistency, measured with
the labelling chapter's script.
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


def rate(rows: list[dict[str, str]], key: str) -> float:
    return sum(int(r[key]) for r in rows) / len(rows) if rows else 0.0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("scores", type=Path)
    args = parser.parse_args(argv)
    with args.scores.open(newline="", encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh))
    h = [r["human_score"] for r in rows]
    j = [r["judge_score"] for r in rows]
    agree = sum(1 for x, y in zip(h, j, strict=True) if x == y)
    false_pass = sum(1 for x, y in zip(h, j, strict=True) if x == "0" and y == "1")
    false_fail = sum(1 for x, y in zip(h, j, strict=True) if x == "1" and y == "0")
    print(
        f"{len(rows)} items: agreement {agree / len(rows):.2f}, kappa {kappa(h, j):.2f}; grader passes what humans fail {false_pass}, fails what humans pass {false_fail}"
    )

    median = sorted(int(r["answer_length"]) for r in rows)[len(rows) // 2]
    short = [r for r in rows if int(r["answer_length"]) < median]
    long_ = [r for r in rows if int(r["answer_length"]) >= median]
    print(f"\n{'group':<24} {'n':>3} {'human pass':>11} {'grader pass':>12}")
    for name, group in (
        ("short answers", short),
        ("long answers", long_),
        ("position A", [r for r in rows if r["position"] == "A"]),
        ("position B", [r for r in rows if r["position"] == "B"]),
    ):
        print(
            f"{name:<24} {len(group):>3} {rate(group, 'human_score'):>11.2f} {rate(group, 'judge_score'):>12.2f}"
        )
    gap_len = (rate(long_, "judge_score") - rate(long_, "human_score")) - (
        rate(short, "judge_score") - rate(short, "human_score")
    )
    if gap_len > 0.15:
        print(
            f"\nlength bias: the grader passes long answers {gap_len:.2f} more often than humans do, relative to short ones; add a length rule to the rubric"
        )
    print(
        "\nThe grader is bounded by the humans' own agreement; validate on a hundred items per task and after every rubric or model change."
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
