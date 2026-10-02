"""Score a search system on a known-item question set.

Reads a question set (CSV with question_id, language, question, expected_id)
and a catalog, runs each question through a search function, and prints
Recall@5, Recall@10, and MRR overall and per language.

The built-in search is a plain keyword overlap over the catalog's `name` and
`definition_long` fields (World Bank indicator schema names). It exists so
that the script runs without any dependencies and gives a baseline. Replace
`search()` with a call to your own search API to score it.

Usage:
    python score_retrieval.py eval_questions.csv example_catalog.csv
"""

from __future__ import annotations

import argparse
import csv
import re
import sys
from collections import defaultdict
from pathlib import Path

TOKEN = re.compile(r"\w+")  # Unicode-aware: words in any script
ID_FIELD = "idno"
TEXT_FIELDS = ("name", "definition_long")
K_VALUES = (5, 10)


def tokens(text: str) -> set[str]:
    return set(TOKEN.findall(text.lower()))


def load_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


def search(question: str, catalog: list[dict[str, str]], k: int = 10) -> list[str]:
    """Return up to k catalog ids ranked by keyword overlap.

    Replace this function with a call to your search system. It must return
    a ranked list of ids.
    """
    query = tokens(question)
    scored = []
    for record in catalog:
        text = tokens(" ".join(record.get(f, "") for f in TEXT_FIELDS))
        overlap = len(query & text)
        if overlap:
            scored.append((overlap, record[ID_FIELD]))
    scored.sort(reverse=True)
    return [rid for _, rid in scored[:k]]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("questions", type=Path)
    parser.add_argument("catalog", type=Path)
    args = parser.parse_args(argv)

    catalog = load_csv(args.catalog)
    questions = [q for q in load_csv(args.questions) if q["expected_id"] != "NONE"]

    hits: dict[int, dict[str, int]] = {k: defaultdict(int) for k in K_VALUES}
    reciprocal_rank: dict[str, float] = defaultdict(float)
    count: dict[str, int] = defaultdict(int)

    print(f"{'id':<5} {'lang':<5} {'rank':>4}  question")
    for q in questions:
        ranked = search(q["question"], catalog, k=max(K_VALUES))
        rank = ranked.index(q["expected_id"]) + 1 if q["expected_id"] in ranked else None
        for group in ("all", q["language"]):
            count[group] += 1
            if rank is not None:
                reciprocal_rank[group] += 1 / rank
                for k in K_VALUES:
                    if rank <= k:
                        hits[k][group] += 1
        print(f"{q['question_id']:<5} {q['language']:<5} {rank or '-':>4}  {q['question']}")

    print()
    header = " ".join(f"{'R@' + str(k):>6}" for k in K_VALUES)
    print(f"{'group':<6} {'n':>3} {header} {'MRR':>6}")
    for group in sorted(count, key=lambda g: (g != "all", g)):
        n = count[group]
        recalls = " ".join(f"{hits[k][group] / n:>6.2f}" for k in K_VALUES)
        print(f"{group:<6} {n:>3} {recalls} {reciprocal_rank[group] / n:>6.2f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
