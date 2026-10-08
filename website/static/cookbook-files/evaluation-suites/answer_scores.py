"""Score a run of generated answers on four measures.

Reads answers (JSON lines: question_id, language, expected_behaviour
answer|decline, expected_values, retrieved_ids, retrieved_values, answer,
cited_ids, declined) and reports:

    numeric accuracy    share of answered questions whose expected value
                        appears in the answer (French decimals and
                        thousands separators are normalized)
    citation validity   share of cited identifiers that were among the
                        retrieved records
    refusal accuracy    share of questions that should be declined and
                        were, and share that should be answered and were
    unsupported numbers share of numbers in answers that are neither a
                        retrieved value nor a year, a date, a "per 1,000"
                        denominator, or an identifier: the working
                        definition of a hallucinated figure

Each measure is also reported per language. Standard library only.

Usage:
    python answer_scores.py answers_run.jsonl

What this does not do: eight answers demonstrate the measures. A number
that matches a retrieved value can still be the wrong value for the
question (the right series, the wrong period), which the grader of the
judges chapter and the human review catch. Units and caveats are not
scored here.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from collections import defaultdict
from pathlib import Path

NUMBER = re.compile(r"(?<![\w.])(\d{1,3}(?:[ ,]\d{3})+|\d+)(?:[.,](\d+))?(?![\w])")
IGNORE = re.compile(
    r"\d{4}-\d{2}-\d{2}|per\s+[\d,. ]+\b|\b[A-Z][A-Z0-9_]{3,}\b"
)  # dates, denominators, identifiers


def numbers(text: str) -> list[float]:
    out = []
    text = IGNORE.sub(" ", text)
    for whole, frac in NUMBER.findall(text):
        whole = re.sub(r"[ ,]", "", whole)
        out.append(float(f"{whole}.{frac}") if frac else float(whole))
    return out


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("answers", type=Path)
    args = parser.parse_args(argv)
    with args.answers.open(encoding="utf-8") as fh:
        rows = [json.loads(line) for line in fh if line.strip()]

    tallies: dict[str, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
    findings: list[str] = []
    for r in rows:
        slices = ["all", f"language={r['language']}"]
        found = numbers(r["answer"])
        if r["expected_behaviour"] == "answer":
            hit = (
                any(abs(v - e) < 1e-9 for e in r["expected_values"] for v in found)
                if not r["declined"]
                else False
            )
            for s in slices:
                tallies[s]["numeric"].append(float(hit))
                tallies[s]["answered_when_expected"].append(float(not r["declined"]))
            if not hit:
                findings.append(
                    f"{r['question_id']}: expected {r['expected_values']}, answer gave {found}"
                )
        else:
            for s in slices:
                tallies[s]["declined_when_expected"].append(float(r["declined"]))
            if not r["declined"]:
                findings.append(
                    f"{r['question_id']}: should have declined, answered {r['answer'][:60]!r}"
                )
        for cid in r["cited_ids"]:
            valid = cid in r["retrieved_ids"]
            for s in slices:
                tallies[s]["citation"].append(float(valid))
            if not valid:
                findings.append(
                    f"{r['question_id']}: cited {cid}, which was not retrieved"
                )
        supported = set(r["retrieved_values"]) | set(r["expected_values"])
        for v in found:
            if 1900 <= v <= 2100 and v.is_integer():
                continue
            ok = any(abs(v - s) < 1e-9 for s in supported)
            for s in slices:
                tallies[s]["unsupported"].append(float(not ok))
            if not ok:
                findings.append(
                    f"{r['question_id']}: number {v:g} is not a retrieved value"
                )

    measures = [
        ("numeric", "numeric accuracy"),
        ("citation", "citation validity"),
        ("declined_when_expected", "refusal accuracy"),
        ("answered_when_expected", "answer rate when expected"),
        ("unsupported", "unsupported numbers"),
    ]
    print(f"{len(rows)} answers\n")
    print(f"{'slice':<14}" + "".join(f"{label:>26}" for _, label in measures))
    for s, t in tallies.items():
        cells = []
        for key, _ in measures:
            vals = t.get(key, [])
            cells.append(
                f"{sum(vals) / len(vals):.2f} (n={len(vals)})" if vals else "-"
            )
        print(f"{s:<14}" + "".join(f"{c:>26}" for c in cells))
    if findings:
        print("\nfindings:")
        for f in findings:
            print(f"  {f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
