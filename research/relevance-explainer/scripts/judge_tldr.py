"""LLM-judge scores for a tagged run's tl;dr, against the run's own explanation and the input.

Reads <run>/<tag>_generations_int8.jsonl (scripts/judge_prep_decode.py keeps the raw answer) and judges the same
random sample as tiny/judge.py (seed 0, --n). Scores: reason (1-5: the tl;dr states the deciding reason for the verdict
that the explanation gives), faithful (1-5: everything in it is supported by the explanation or the input), and
language_ok.

  LLM_BASE_URL=http://localhost:8000/v1,http://localhost:8001/v1 \
  LLM_EXTRA='{"chat_template_kwargs":{"enable_thinking":false}}' python scripts/judge_tldr.py runs/fmt-s-C --tag realnb
"""
import argparse
import json
import random
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from relevance_explainer.claims import tldr_of
from relevance_explainer.synth import llm

SYSTEM = """You grade a one-sentence summary (tl;dr) of an explanation written by a system for a data-catalog search result. You are given the system input (search, item metadata, checks), the verdict with its explanation, and the tl;dr.

Score:
- reason (1-5): 5 = the tl;dr states the deciding reason for the verdict that the explanation gives; 3 = it states a reason but not the main one, or only vaguely; 1 = it states no reason, or a reason that contradicts the explanation or the verdict.
- faithful (1-5): 5 = everything in the tl;dr is supported by the explanation or the input; 3 = a doubtful detail; 1 = mostly invented or wrong.
- language_ok: true if the tl;dr is written in the language named on the "Respond in:" line.

Return only a JSON object: {"reason": int, "faithful": int, "language_ok": bool}"""

ap = argparse.ArgumentParser()
ap.add_argument("run")
ap.add_argument("--tag", default="realnb")
ap.add_argument("--n", type=int, default=800)
ap.add_argument("--model", default="qwen3.8-27b")
ap.add_argument("--workers", type=int, default=48)
a = ap.parse_args()
run = Path(a.run)
rows = [json.loads(line) for line in open(run / f"{a.tag}_generations_int8.jsonl")]
rows = random.Random(0).sample(rows, min(a.n, len(rows)))


def one(r: dict) -> dict | None:
    t = tldr_of(r.get("raw", ""))
    if not t:
        return {"missing": True}
    msg = f"INPUT:\n{r['input']}\n\nVERDICT AND EXPLANATION:\n{r['generated']}\n\nTL;DR:\n{t}"
    try:
        out, _ = llm.chat(a.model, SYSTEM, msg, temperature=0.0, max_tokens=80)
        j = llm.parse_json(out)
        return {"reason": int(j["reason"]), "faithful": int(j["faithful"]), "language_ok": bool(j["language_ok"])}
    except Exception:
        return None


with ThreadPoolExecutor(a.workers) as ex:
    res = list(ex.map(one, rows))
ok = [s for s in res if s and "missing" not in s]
n = len(ok)
summary = {"n": n, "missing": sum(bool(s and "missing" in s) for s in res), "failed": sum(s is None for s in res),
           "reason": sum(s["reason"] for s in ok) / n, "reason_ge4": sum(s["reason"] >= 4 for s in ok) / n,
           "faithful": sum(s["faithful"] for s in ok) / n, "faithful_ge4": sum(s["faithful"] >= 4 for s in ok) / n,
           "language_ok": sum(s["language_ok"] for s in ok) / n}
(run / f"judge_tldr_{a.tag}.json").write_text(json.dumps({"summary": summary, "items": res}, indent=1))
print(run.name, {k: round(v, 3) if isinstance(v, float) else v for k, v in summary.items()})
