"""Score a search system on a known-item question set.

Reads a question set (CSV with question_id, language, question, expected_id)
and a catalog, runs each question through a search function, and prints
Recall@5, Recall@10, and MRR overall and per language.

The built-in search is a plain keyword overlap over title and description. It
exists so that the script runs without any dependencies and gives a baseline.
Replace `search()` with a call to your own search API to score it.

Usage:
    python score_retrieval.py eval_questions.csv example_catalog.csv
"""

import argparse
import csv
import re
from collections import defaultdict

TOKEN = re.compile(r"[a-zà-ÿ0-9]+", re.IGNORECASE)


def tokens(text):
    return set(TOKEN.findall((text or "").lower()))


def load_catalog(path):
    with open(path, newline="", encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


def search(question, catalog, k=10):
    """Return up to k catalog ids ranked by keyword overlap.

    Replace this function with a call to your search system. It must return
    a ranked list of ids.
    """
    q = tokens(question)
    scored = []
    for rec in catalog:
        text = tokens(rec["title"] + " " + rec.get("description", ""))
        overlap = len(q & text)
        if overlap:
            scored.append((overlap, rec["id"]))
    scored.sort(reverse=True)
    return [rid for _, rid in scored[:k]]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("questions")
    parser.add_argument("catalog")
    args = parser.parse_args()

    catalog = load_catalog(args.catalog)
    with open(args.questions, newline="", encoding="utf-8") as fh:
        questions = [q for q in csv.DictReader(fh) if q["expected_id"] != "NONE"]

    hits5 = defaultdict(int)
    hits10 = defaultdict(int)
    rr = defaultdict(float)
    n = defaultdict(int)

    print(f"{'id':<5} {'lang':<5} {'rank':>4}  question")
    for q in questions:
        ranked = search(q["question"], catalog)
        rank = ranked.index(q["expected_id"]) + 1 if q["expected_id"] in ranked else None
        for key in ("all", q["language"]):
            n[key] += 1
            if rank is not None:
                rr[key] += 1 / rank
                if rank <= 5:
                    hits5[key] += 1
                if rank <= 10:
                    hits10[key] += 1
        print(f"{q['question_id']:<5} {q['language']:<5} {rank or '-':>4}  {q['question']}")

    print()
    print(f"{'group':<6} {'n':>3} {'R@5':>6} {'R@10':>6} {'MRR':>6}")
    for key in sorted(n, key=lambda k: (k != "all", k)):
        print(
            f"{key:<6} {n[key]:>3} {hits5[key] / n[key]:>6.2f} "
            f"{hits10[key] / n[key]:>6.2f} {rr[key] / n[key]:>6.2f}"
        )


if __name__ == "__main__":
    main()
