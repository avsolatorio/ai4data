"""Map free-text keywords to a controlled vocabulary.

Reads keywords as curators or models wrote them (CSV: record_id, keyword)
and a vocabulary (CSV: preferred_label, uri, alternates separated by ";")
and assigns each keyword a preferred label by exact match on the preferred
label or an alternate, or by similarity above a threshold (difflib ratio),
or marks it unmapped. Prints the mapping and the unmapped keywords, which
are the candidates for new alternates or new concepts. Standard library
only.

The vocabulary format is the minimum of a SKOS concept scheme: a preferred
label, a URI, and alternate labels. A full scheme adds definitions,
broader and narrower relations, and labels per language.

Usage:
    python map_vocabulary.py keywords_freetext.csv vocabulary.csv

What this does not do: string matching maps known forms. A keyword that
names a concept in different words ("hunger" for undernourishment) is
unmapped until an alternate is added or a semantic match (embeddings) is
used, and a language model can propose the mapping for a curator to
confirm.
"""

from __future__ import annotations

import argparse
import csv
import difflib
import re
import sys
from pathlib import Path

WORD = re.compile(r"[a-z0-9]+")


def norm(text: str) -> str:
    return " ".join(WORD.findall(text.lower()))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("keywords", type=Path)
    parser.add_argument("vocabulary", type=Path)
    parser.add_argument("--threshold", type=float, default=0.85)
    args = parser.parse_args(argv)

    with args.vocabulary.open(newline="", encoding="utf-8") as fh:
        concepts = list(csv.DictReader(fh))
    index: dict[str, str] = {}
    for c in concepts:
        index[norm(c["preferred_label"])] = c["preferred_label"]
        for alt in (c.get("alternates") or "").split(";"):
            if alt.strip():
                index[norm(alt)] = c["preferred_label"]
    with args.keywords.open(newline="", encoding="utf-8") as fh:
        keywords = list(csv.DictReader(fh))

    counts = {"exact": 0, "fuzzy": 0, "none": 0}
    unmapped: list[str] = []
    print(f"{'match':<6} {'score':>5}  {'record':<22} {'keyword':<34} -> preferred label")
    for k in keywords:
        text = norm(k["keyword"])
        if text in index:
            kind, score, label = "exact", 1.0, index[text]
        else:
            best = max(((difflib.SequenceMatcher(None, text, form).ratio(), label) for form, label in index.items()), default=(0.0, ""))
            if best[0] >= args.threshold:
                kind, score, label = "fuzzy", best[0], best[1]
            else:
                kind, score, label = "none", 0.0, "-"
                unmapped.append(f"{k['record_id']}: {k['keyword']!r}")
        counts[kind] += 1
        print(f"{kind:<6} {score:>5.2f}  {k['record_id']:<22} {k['keyword']!r:<34} -> {label}")
    print(f"\n{len(keywords)} keywords: " + ", ".join(f"{k} {v}" for k, v in counts.items()))
    if unmapped:
        print("unmapped (add an alternate, add a concept, or let a model propose a mapping for review):")
        for u in unmapped:
            print(f"  {u}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
