"""Score variable-level search on a known-variable question set.

Users of microdata look for variables ("age of household head"), and most
often across several surveys. This script runs each question in a question
set through a search function over the data dictionary and reports Recall@5
and MRR, overall and per language.

The built-in search is a keyword overlap over the label, the question text,
and the concept of each variable. It exists so that the script runs with no
dependencies and gives a baseline. Replace `search()` with a call to the
catalog's variable search to score it.

Usage:
    python score_variable_search.py variable_questions.csv lfs_2025q2_dictionary.csv

What this does not do: thirteen questions over one dictionary give a
demonstration. A usable set has 30 to 50 questions per language and covers
several surveys, so that the score reflects cross-survey search.
"""

from __future__ import annotations

import argparse
import csv
import re
import sys
from collections import defaultdict
from pathlib import Path

TOKEN = re.compile(r"\w+")
TEXT_FIELDS = ("label", "question", "concept")
K = 5


def tokens(text: str) -> set[str]:
    return set(TOKEN.findall(text.lower()))


def load_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


def search(question: str, dictionary: list[dict[str, str]], k: int = K) -> list[str]:
    """Return up to k variable names ranked by keyword overlap. Replace with the real search."""
    query = tokens(question)
    scored = []
    for var in dictionary:
        text = tokens(" ".join(var.get(f, "") for f in TEXT_FIELDS) + " " + var["name"].replace("_", " "))
        overlap = len(query & text)
        if overlap:
            scored.append((overlap, var["name"]))
    scored.sort(reverse=True)
    seen: list[str] = []
    for _, name in scored:
        if name not in seen:
            seen.append(name)
    return seen[:k]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("questions", type=Path)
    parser.add_argument("dictionary", type=Path)
    args = parser.parse_args(argv)

    dictionary = load_csv(args.dictionary)
    questions = [q for q in load_csv(args.questions) if q["expected_name"] != "NONE"]
    hits: dict[str, int] = defaultdict(int)
    rr: dict[str, float] = defaultdict(float)
    n: dict[str, int] = defaultdict(int)

    print(f"{'id':<4} {'lang':<5} {'rank':>4}  question")
    for q in questions:
        ranked = search(q["question"], dictionary)
        rank = ranked.index(q["expected_name"]) + 1 if q["expected_name"] in ranked else None
        for group in ("all", q["language"]):
            n[group] += 1
            if rank is not None:
                hits[group] += 1
                rr[group] += 1 / rank
        print(f"{q['question_id']:<4} {q['language']:<5} {rank or '-':>4}  {q['question']}")

    print()
    print(f"{'group':<6} {'n':>3} {'R@5':>6} {'MRR':>6}")
    for group in sorted(n, key=lambda g: (g != "all", g)):
        print(f"{group:<6} {n[group]:>3} {hits[group] / n[group]:>6.2f} {rr[group] / n[group]:>6.2f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
