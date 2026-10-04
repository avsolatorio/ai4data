"""Second and third verdicts for a realistic evaluation set: the teacher with thinking on, and Gemma 4.

Both use the realistic narration prompt on the full-format input. Writes examples_labeled.jsonl with
labels = {stored, thinking, gemma4}, the majority label and whether all three agree.

  python -m relevance_explainer.synth.label_consensus --data data/real_test
"""
from __future__ import annotations

import argparse
import json
import re
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from ..distill.teacher_checks import post
from . import llm
from .realistic import NARRATE_V2

V = ("relevant", "partly_relevant", "irrelevant")


def verdict_from(text: str) -> str | None:
    text = (text or "").split("</think>")[-1]
    try:
        v = llm.parse_json(text)["verdict"]
    except Exception:
        m = re.search(r'"verdict"\s*:\s*"(\w+)"', text)
        v = m.group(1) if m else None
    return v if v in V else None


def thinking(base: str, text: str) -> str | None:
    body = {"model": "qwen3.8-27b", "max_tokens": 2400, "temperature": 0.3,
            "messages": [{"role": "system", "content": NARRATE_V2}, {"role": "user", "content": text}],
            "chat_template_kwargs": {"enable_thinking": True}}
    try:
        return verdict_from(post(base, body)["choices"][0]["message"]["content"])
    except Exception:
        return None


def gemma(base: str, text: str) -> str | None:
    body = {"model": "gemma-4-31b", "max_tokens": 450, "temperature": 0.0,
            "messages": [{"role": "system", "content": NARRATE_V2}, {"role": "user", "content": text}],
            "chat_template_kwargs": {"enable_thinking": False}}
    try:
        return verdict_from(post(base, body)["choices"][0]["message"]["content"])
    except Exception:
        return None


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--teacher-urls", default="http://localhost:8000/v1,http://localhost:8001/v1")
    ap.add_argument("--gemma-url", default="http://localhost:8002/v1")
    a = ap.parse_args()
    d = Path(a.data)
    rows = [json.loads(l) for l in (d / "examples.jsonl").read_text().splitlines()]
    urls = a.teacher_urls.split(",")
    with ThreadPoolExecutor(48) as ex:
        th = list(ex.map(lambda k: thinking(urls[k % len(urls)], rows[k]["input_v2"]), range(len(rows))))
    with ThreadPoolExecutor(16) as ex:
        gm = list(ex.map(lambda r: gemma(a.gemma_url, r["input_v2"]), rows))
    with (d / "examples_labeled.jsonl").open("w") as f:
        for r, t, g in zip(rows, th, gm):
            labs = {"stored": r["verdict"], "thinking": t, "gemma4": g}
            got = [v for v in labs.values() if v]
            top = Counter(got).most_common(1)[0]
            r = {**r, "labels": labs, "majority": top[0] if top[1] >= 2 else None, "unanimous": len(got) == 3 and len(set(got)) == 1}
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    n = len(rows)
    print(f"labeled {n}: thinking agrees {sum(t == r['verdict'] for r, t in zip(rows, th)) / n:.3f}, "
          f"gemma agrees {sum(g == r['verdict'] for r, g in zip(rows, gm)) / n:.3f}, "
          f"unanimous {sum(1 for r, t, g in zip(rows, th, gm) if t and g and t == g == r['verdict']) / n:.3f}")


if __name__ == "__main__":
    main()
