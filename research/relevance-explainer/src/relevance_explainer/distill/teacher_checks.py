"""Teacher-side distillation tools (vLLM OpenAI-compatible servers).

  soft   record the teacher's probabilities over the three verdicts for each training input
  think  re-label a sample with thinking mode on and compare with the stored verdicts (label noise)

Servers are given as TEACHER_URLS, a comma-separated list such as http://localhost:8000/v1,http://localhost:8001/v1.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import random
import re
import urllib.request
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from ..synth.generate import NARRATE_SYSTEM
from ..synth.llm import parse_json

MODEL = os.environ.get("TEACHER_MODEL", "qwen3.8-27b")
CLASSES = ["relevant", "partly_relevant", "irrelevant"]


def urls() -> list[str]:
    return os.environ.get("TEACHER_URLS", "http://localhost:8000/v1").split(",")


def post(base: str, body: dict, timeout: int = 600) -> dict:
    req = urllib.request.Request(base + "/chat/completions", json.dumps(body).encode(), {"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.load(r)


def strip_similarity(text: str) -> str:
    """Teacher prompts are the original inputs, which do not contain the retrieval-similarity line."""
    return "\n".join(l for l in text.split("\n") if not l.startswith("Retrieval similarity:"))


def class_of(token: str) -> str | None:
    t = token.strip().strip('"').lower()
    if t.startswith("part"):  # "partly_relevant" is tokenized as "part" + "ly_relevant"
        return "partly_relevant"
    if t.startswith("ir"):
        return "irrelevant"
    if t.startswith("relev"):
        return "relevant"
    return None


def verdict_probs(base: str, text: str) -> list[float] | None:
    """Probabilities of the first verdict token after the prefill '{"verdict": "' of the narrate prompt."""
    body = {"model": MODEL, "max_tokens": 1, "temperature": 0.0, "logprobs": True, "top_logprobs": 20,
            "messages": [{"role": "system", "content": NARRATE_SYSTEM}, {"role": "user", "content": strip_similarity(text)},
                         {"role": "assistant", "content": '{"verdict": "'}],
            "continue_final_message": True, "add_generation_prompt": False,
            "chat_template_kwargs": {"enable_thinking": False}}
    top = post(base, body)["choices"][0]["logprobs"]["content"][0]["top_logprobs"]
    mass = {c: 0.0 for c in CLASSES}
    for t in top:
        c = class_of(t["token"])
        if c:
            mass[c] += math.exp(t["logprob"])
    total = sum(mass.values())
    return [mass[c] / total for c in CLASSES] if total > 0.5 else None


def cmd_soft(a) -> None:
    rows = [json.loads(l) for l in Path(a.data, f"{a.split}.jsonl").read_text().splitlines()]
    out = Path(a.data, f"soft_{a.split}.jsonl")
    done = {json.loads(l)["i"] for l in out.read_text().splitlines()} if out.exists() else set()
    todo = [i for i in range(len(rows)) if i not in done][: a.limit or None]
    servers = urls()

    def one(i: int):
        try:
            return i, verdict_probs(servers[i % len(servers)], rows[i]["input"])
        except Exception as e:
            print("skip", i, type(e).__name__, str(e)[:80])
            return i, None

    n = 0
    with ThreadPoolExecutor(a.workers) as ex, out.open("a") as f:
        for i, p in ex.map(one, todo):
            if p:
                f.write(json.dumps({"i": i, "p": [round(x, 4) for x in p]}) + "\n")
                n += 1
    print(f"soft labels: +{n}, {len(done) + n} total of {len(rows)}")


def cmd_softstats(a) -> None:
    rows = [json.loads(l) for l in Path(a.data, f"{a.split}.jsonl").read_text().splitlines()]
    soft = {d["i"]: d["p"] for d in map(json.loads, Path(a.data, f"soft_{a.split}.jsonl").read_text().splitlines())}
    agree = conf = 0
    lowconf = Counter()
    for i, p in soft.items():
        y = CLASSES.index(rows[i]["verdict"])
        agree += int(max(range(3), key=lambda k: p[k]) == y)
        conf += p[y]
        lowconf[rows[i]["verdict"]] += int(p[y] < 0.6)
    n = len(soft)
    print(f"n {n}; argmax(soft) == stored verdict {agree / n:.3f}; mean prob of stored verdict {conf / n:.3f}")
    print("stored verdict with teacher prob < 0.6:", {c: f"{lowconf[c]}" for c in CLASSES}, f"({sum(lowconf.values()) / n:.3f} of all)")


def think_one(base: str, text: str) -> str | None:
    body = {"model": MODEL, "max_tokens": 2200, "temperature": 0.3,
            "messages": [{"role": "system", "content": NARRATE_SYSTEM}, {"role": "user", "content": strip_similarity(text)}],
            "chat_template_kwargs": {"enable_thinking": True}}
    try:
        content = post(base, body)["choices"][0]["message"]["content"] or ""
    except Exception as e:  # over-long prompt or server error: leave this example out of the comparison
        print("skip:", type(e).__name__, str(e)[:60])
        return None
    content = content.split("</think>")[-1]
    try:
        v = parse_json(content)["verdict"]
        return v if v in CLASSES else None
    except Exception:
        m = re.search(r'"verdict"\s*:\s*"(\w+)"', content)
        return m.group(1) if m and m.group(1) in CLASSES else None


def cmd_think(a) -> None:
    rows = [json.loads(l) for l in Path(a.data, f"{a.split}.jsonl").read_text().splitlines()]
    sample = random.Random(0).sample(range(len(rows)), min(a.n, len(rows)))
    servers = urls()
    with ThreadPoolExecutor(a.workers) as ex:
        res = list(ex.map(lambda k: think_one(servers[k % len(servers)], rows[sample[k]]["input"]), range(len(sample))))
    pairs = [(rows[i]["verdict"], v) for i, v in zip(sample, res) if v]
    agree = sum(s == t for s, t in pairs) / len(pairs)
    print(f"thinking-mode verdicts: {len(pairs)} parsed of {len(sample)}; agree with stored (no-thinking) verdict {agree:.3f}")
    print("stored -> thinking:", dict(sorted(Counter(pairs).items())))


def main() -> None:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    for name, fn in (("soft", cmd_soft), ("softstats", cmd_softstats), ("think", cmd_think)):
        p = sub.add_parser(name)
        p.add_argument("--data", required=True)
        p.add_argument("--split", default="train")
        p.add_argument("--workers", type=int, default=256)
        p.add_argument("--limit", type=int, default=0)
        p.add_argument("--n", type=int, default=500)
        p.set_defaults(fn=fn)
    a = ap.parse_args()
    a.fn(a)


if __name__ == "__main__":
    main()
