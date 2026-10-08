"""Score model suggestions against curator decisions.

Reads the suggestions a model or a review pipeline produced (JSON lines:
record_id, field, source, current, suggested, issue_category,
issue_severity, model_confidence) and the decisions curators recorded on
them (CSV: record_id, field, decision accept|edit|reject, final_text,
reason, reviewer, date), and reports per field and per source:

    n          suggestions decided
    accepted   taken as proposed
    edited     taken after a change; the similarity between the suggestion
               and the final text shows how much was changed
    rejected   with the curators' reasons listed

The acceptance rate (accepted plus edited over decided) per field and per
source is the measure that chapter 6 tracks over time. Standard library
only.

Usage:
    python score_suggestions.py suggestions_example.jsonl decisions_example.csv

What this does not do: ten suggestions give a demonstration; a measure
needs a period's decisions, a few hundred, with the reasons coded into an
error taxonomy. The script counts; it does not judge whether curators were
right.
"""

from __future__ import annotations

import argparse
import csv
import difflib
import json
import sys
from collections import defaultdict
from pathlib import Path


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("suggestions", type=Path)
    parser.add_argument("decisions", type=Path)
    args = parser.parse_args(argv)

    with args.suggestions.open(encoding="utf-8") as fh:
        suggestions = {(s["record_id"], s["field"]): s for s in (json.loads(line) for line in fh if line.strip())}
    with args.decisions.open(newline="", encoding="utf-8") as fh:
        decisions = list(csv.DictReader(fh))

    groups = {"field": defaultdict(lambda: defaultdict(int)), "source": defaultdict(lambda: defaultdict(int))}
    similarities: list[float] = []
    rejections: list[str] = []
    undecided = set(suggestions)
    for d in decisions:
        key = (d["record_id"], d["field"])
        s = suggestions.get(key)
        if s is None:
            continue
        undecided.discard(key)
        verdict = d["decision"].strip().lower()
        for by, label in (("field", s["field"]), ("source", s.get("source", "?"))):
            groups[by][label]["n"] += 1
            groups[by][label][verdict] += 1
        if verdict == "edit":
            similarities.append(difflib.SequenceMatcher(None, s["suggested"], d.get("final_text", "")).ratio())
        if verdict == "reject":
            rejections.append(f"{d['record_id']}/{d['field']} ({s.get('source', '?')}): {d.get('reason', '')}")

    decided = sum(g["n"] for g in groups["field"].values())
    print(f"{len(suggestions)} suggestions, {decided} decided, {len(undecided)} undecided")
    for by in ("field", "source"):
        print(f"\n{'by ' + by:<28} {'n':>3} {'accept':>7} {'edit':>5} {'reject':>7} {'rate':>6}")
        for label, g in sorted(groups[by].items(), key=lambda kv: -kv[1]["n"]):
            rate = (g["accept"] + g["edit"]) / g["n"] if g["n"] else 0.0
            print(f"{label:<28} {g['n']:>3} {g['accept']:>7} {g['edit']:>5} {g['reject']:>7} {rate:>6.2f}")
    if similarities:
        print(f"\nedited suggestions: mean similarity to final text {sum(similarities) / len(similarities):.2f}")
    if rejections:
        print("\nrejected, with reasons:")
        for r in rejections:
            print(f"  {r}")
    print("\nThe acceptance rate counts curator decisions; it does not judge them.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
