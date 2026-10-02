"""LLM-judge scores for generated narratives, with the teacher's narratives on the same inputs as a reference.

Uses the OpenAI-compatible endpoint configured through LLM_BASE_URL and LLM_EXTRA (see synth.llm).
  LLM_BASE_URL=http://localhost:8000/v1 LLM_EXTRA='{"chat_template_kwargs":{"enable_thinking":false}}' \
  python -m relevance_explainer.tiny.judge --run runs/t-sim --prec int8 --n 300 --model qwen3.8-27b
"""
from __future__ import annotations

import argparse
import json
import random
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from ..synth import llm

SYSTEM = """You grade a short explanation written by a system for a data-catalog search result. You are given the system input (query, filters, retrieval similarity, item metadata, filter check) and the explanation with its verdict line.

Score:
- faithful (1-5): 5 = every factual claim is supported by the input; 3 = some unsupported or doubtful detail; 1 = mostly invented, wrong, or contradicting the input.
- coherent (1-5): 5 = fluent and sensible; 3 = understandable but awkward or partly repetitive; 1 = garbled, nonsense words, or loops.
- consistent: true if the explanation argues for the verdict on its first line (relevant, partly_relevant or irrelevant), false if it argues against it or is contradictory.
- language_ok: true if the explanation is written in the language named on the "Respond in:" line.

Return only a JSON object: {"faithful": int, "coherent": int, "consistent": bool, "language_ok": bool}"""


def judge(model: str, inp: str, text: str) -> dict | None:
    try:
        out, _ = llm.chat(model, SYSTEM, f"INPUT:\n{inp}\n\nEXPLANATION:\n{text}", temperature=0.0, max_tokens=120)
        j = llm.parse_json(out)
        return {"faithful": int(j["faithful"]), "coherent": int(j["coherent"]),
                "consistent": bool(j["consistent"]), "language_ok": bool(j["language_ok"])}
    except Exception:
        return None


def summarize(scores: list[dict]) -> dict:
    n = len(scores)
    return {"n": n, "faithful": sum(s["faithful"] for s in scores) / n, "coherent": sum(s["coherent"] for s in scores) / n,
            "faithful_ge4": sum(s["faithful"] >= 4 for s in scores) / n, "consistent": sum(s["consistent"] for s in scores) / n,
            "language_ok": sum(s["language_ok"] for s in scores) / n}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--prec", default="int8")
    ap.add_argument("--tag", default="test", help="generations file prefix: <tag>_generations_<prec>.jsonl")
    ap.add_argument("--n", type=int, default=300)
    ap.add_argument("--model", default="qwen3.8-27b")
    ap.add_argument("--workers", type=int, default=32)
    ap.add_argument("--language", default=None, help="judge only generations for this query language")
    ap.add_argument("--out-suffix", default="", help="appended to the judge file name")
    a = ap.parse_args()

    run = Path(a.run)
    rows = [json.loads(l) for l in (run / f"{a.tag}_generations_{a.prec}.jsonl").read_text().splitlines()]
    if a.language:
        rows = [r for r in rows if r["language"] == a.language]
    rows = random.Random(0).sample(rows, min(a.n, len(rows)))
    jobs = [(i, who, r["input"], r[key]) for i, r in enumerate(rows) for who, key in (("student", "generated"), ("teacher", "reference"))]
    with ThreadPoolExecutor(a.workers) as ex:
        res = list(ex.map(lambda j: judge(a.model, j[2], j[3]), jobs))

    by: dict[str, list[dict]] = defaultdict(list)
    by_lang: dict[tuple[str, str], list[dict]] = defaultdict(list)
    for (i, who, _, _), s in zip(jobs, res):
        if s:
            by[who].append(s)
            by_lang[(who, rows[i]["language"])].append(s)
    out = {who: summarize(v) for who, v in by.items()}
    out["by_language"] = {f"{who}/{lang}": summarize(v) for (who, lang), v in sorted(by_lang.items()) if len(v) >= 15}
    (run / f"judge_{a.tag}_{a.prec}{a.out_suffix}.json").write_text(json.dumps(out, indent=2))
    for who in ("teacher", "student"):
        print(who, {k: round(v, 3) if isinstance(v, float) else v for k, v in out[who].items()})
    for k, v in out["by_language"].items():
        print(" ", k, "faithful", round(v["faithful"], 2), "coherent", round(v["coherent"], 2), "n", v["n"])


if __name__ == "__main__":
    main()
