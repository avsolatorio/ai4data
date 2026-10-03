"""Pairwise preference between two runs' explanations for the same test inputs, judged in random order.

Faithfulness alone rewards saying less; this asks which explanation would help a searcher more, given that both
must be accurate. Uses the endpoint in LLM_BASE_URL.

  LLM_BASE_URL=http://localhost:8002/v1 python -m relevance_explainer.tiny.pairwise runs/f5-dpo runs/f5-grpo \
      --model gemma-4-31b --n 800
"""
from __future__ import annotations

import argparse
import json
import random
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from ..synth import llm

SYSTEM = """You compare two explanations written for the same data-catalog search result. You are given the system input (query, filters, retrieval similarity, item metadata, filter check) and explanations A and B, each with a verdict line.

Pick the explanation that would help the person searching more: it is accurate with respect to the input, it tells them what the item contains and how it relates to their query and filters, and it is clear. An explanation that is shorter but leaves out useful facts is not better; an explanation with an unsupported or wrong claim is worse. Answer "tie" only if they are equally useful.

Return only a JSON object: {"better": "A" | "B" | "tie"}"""


def ask(model: str, inp: str, a: str, b: str) -> str | None:
    try:
        out, _ = llm.chat(model, SYSTEM, f"INPUT:\n{inp}\n\nEXPLANATION A:\n{a}\n\nEXPLANATION B:\n{b}", temperature=0.0, max_tokens=40)
        v = str(llm.parse_json(out)["better"]).strip().upper()
        return v if v in ("A", "B", "TIE") else None
    except Exception:
        return None


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("a")
    ap.add_argument("b")
    ap.add_argument("--tag", default="f2test")
    ap.add_argument("--tag-b", default=None, help="generations tag for run b (default: --tag)")
    ap.add_argument("--prec", default="int8")
    ap.add_argument("--n", type=int, default=800)
    ap.add_argument("--model", default="gemma-4-31b")
    ap.add_argument("--workers", type=int, default=32)
    a = ap.parse_args()
    ra = [json.loads(l) for l in (Path(a.a) / f"{a.tag}_generations_{a.prec}.jsonl").read_text().splitlines()]
    tag_b = a.tag_b or a.tag
    rb = [json.loads(l) for l in (Path(a.b) / f"{tag_b}_generations_{a.prec}.jsonl").read_text().splitlines()]
    assert len(ra) == len(rb) and all(x["input"] == y["input"] for x, y in zip(ra, rb))
    idx = random.Random(0).sample(range(len(ra)), min(a.n, len(ra)))  # the judge's usual seeded sample
    rng = random.Random(1)
    swap = {i: rng.random() < 0.5 for i in idx}
    jobs = [i for i in idx if ra[i]["generated"].strip() != rb[i]["generated"].strip()]

    def one(i: int) -> str | None:
        x, y = (rb[i]["generated"], ra[i]["generated"]) if swap[i] else (ra[i]["generated"], rb[i]["generated"])
        v = ask(a.model, ra[i]["input"], x, y)
        if v is None or v == "TIE":
            return v and "tie"
        return ("b" if v == "A" else "a") if swap[i] else ("a" if v == "A" else "b")

    with ThreadPoolExecutor(a.workers) as ex:
        res = dict(zip(jobs, ex.map(one, jobs)))
    c = Counter(v for v in res.values() if v)
    by = defaultdict(Counter)
    for i, v in res.items():
        if v:
            by[ra[i]["language"]][v] += 1
    out = {"a": a.a, "b": a.b, "judge": a.model, "n_sampled": len(idx), "identical": len(idx) - len(jobs),
           "a_better": c["a"], "b_better": c["b"], "tie": c["tie"], "failed": sum(v is None for v in res.values()),
           "by_language": {k: dict(v) for k, v in sorted(by.items())}}
    name = f"pairwise_{Path(a.a).name}-{a.tag}_vs_{Path(a.b).name}-{tag_b}_{a.model}.json"
    (Path(a.b) / name).write_text(json.dumps(out, indent=2))
    print(json.dumps({k: v for k, v in out.items() if k != "by_language"}))


if __name__ == "__main__":
    main()
