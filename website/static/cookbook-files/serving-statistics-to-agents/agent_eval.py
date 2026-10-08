"""Score an agent's logged runs against a question set.

Reads a question set (CSV: question_id, language, question, expected_tools
separated by ";", expected_series, expected_value, expected_behaviour
answer|decline) and the traces an agent harness logged (JSON lines:
question_id, tools_called, series, answer, declined, latency_s), and
reports per question and in total:

    tools     every expected tool was called
    series    the series the agent used is the expected one
    value     the expected value appears in the answer (within tolerance)
    cite      the answer contains a source URL
    behave    the agent answered when it should and declined when it should

Exit status is 0 when every check passes, 1 otherwise. Standard library
only.

What this does not do: it scores what the trace records. Whether the
answer's wording is correct beyond the value and the citation, whether
caveats were repeated, and whether the tool calls were efficient are
judged by a reviewer on a sample. Eight questions give a demonstration; a
suite has 30 to 50 per language.

Usage:
    python agent_eval.py agent_questions.csv agent_traces.jsonl
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import sys
from pathlib import Path

NUMBER = re.compile(r"-?\d[\d,]*(?:[.,]\d+)?")
URL = re.compile(r"https?://\S+")


def numbers_in(text: str) -> list[float]:
    out = []
    for raw in NUMBER.findall(text):
        cleaned = raw.replace(" ", "")
        # "6,3" in French is 6.3; "12,500" is 12500
        if re.fullmatch(r"-?\d+,\d{1,2}", cleaned):
            cleaned = cleaned.replace(",", ".")
        else:
            cleaned = cleaned.replace(",", "")
        try:
            out.append(float(cleaned))
        except ValueError:
            continue
    return out


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("questions", type=Path)
    parser.add_argument("traces", type=Path)
    parser.add_argument("--tolerance", type=float, default=0.05)
    args = parser.parse_args(argv)

    with args.questions.open(newline="", encoding="utf-8") as fh:
        questions = list(csv.DictReader(fh))
    with args.traces.open(encoding="utf-8") as fh:
        traces = {t["question_id"]: t for t in (json.loads(line) for line in fh if line.strip())}

    checks = ("tools", "series", "value", "cite", "behave")
    totals = dict.fromkeys(checks, 0)
    applicable = dict.fromkeys(checks, 0)
    failures: list[str] = []
    latencies: list[float] = []
    print(f"{'id':<4} {'lang':<5} {'tools':<6} {'series':<7} {'value':<6} {'cite':<5} {'behave':<7} question")
    for q in questions:
        t = traces.get(q["question_id"])
        if t is None:
            failures.append(f"{q['question_id']}: no trace")
            continue
        latencies.append(float(t.get("latency_s", 0)))
        expected_tools = [x for x in q["expected_tools"].split(";") if x]
        should_decline = q["expected_behaviour"] == "decline"
        results = {}
        results["tools"] = all(tool in t.get("tools_called", []) for tool in expected_tools)
        results["behave"] = bool(t.get("declined")) == should_decline
        if should_decline:
            results["series"] = results["value"] = results["cite"] = None
        else:
            results["series"] = (t.get("series") == q["expected_series"]) if q["expected_series"] else None
            if q["expected_value"]:
                target = float(q["expected_value"])
                results["value"] = any(
                    abs(n - target) <= args.tolerance for n in numbers_in(t.get("answer", ""))
                )
            else:
                results["value"] = None
            results["cite"] = bool(URL.search(t.get("answer", "")))
        for c in checks:
            if results[c] is None:
                continue
            applicable[c] += 1
            if results[c]:
                totals[c] += 1
            else:
                failures.append(f"{q['question_id']}: {c} failed ({q.get('note') or q['question']})")
        cell = lambda v: "-" if v is None else ("ok" if v else "FAIL")
        print(
            f"{q['question_id']:<4} {q['language']:<5} {cell(results['tools']):<6} {cell(results['series']):<7} "
            f"{cell(results['value']):<6} {cell(results['cite']):<5} {cell(results['behave']):<7} {q['question']}"
        )

    print()
    for c in checks:
        if applicable[c]:
            print(f"{c:<7} {totals[c]}/{applicable[c]}  {totals[c] / applicable[c]:.2f}")
    if latencies:
        latencies.sort()
        print(f"latency median {latencies[len(latencies) // 2]:.1f}s, max {latencies[-1]:.1f}s")
    if failures:
        print("\nfailures:")
        for f in failures:
            print(f"  {f}")
    print("\nNot checked here: wording, caveats, and efficiency of the tool calls; a reviewer samples those.")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
