"""One-sentence rationale format for the hybrid output (structured filter display plus a short model-written sentence).

The teacher writes one sentence per training input for the already stored verdict. Filter results are left to a
deterministic display, so the sentence covers only what the item contains and how it relates to the query.

  TEACHER_URLS=http://localhost:8000/v1 python -m relevance_explainer.distill.short_format \
      --data data/f1 --out data/f1short --old-out data/f1old10k --n 10000
"""
from __future__ import annotations

import argparse
import json
import os
import random
import re
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from ..synth.llm import parse_json
from .teacher_checks import MODEL, post, strip_similarity, urls

SYSTEM = """You write the one-sentence rationale shown next to a data catalog search result. You are given the user's query, the filters, the item metadata and the verdict (relevant, partly_relevant or irrelevant).

Write ONE sentence of at most 25 words, in the language named on the "Respond in:" line, that says what the item contains and how it relates to what the query asks. Support the verdict: for partly_relevant say which part of the query the item addresses and which part it does not; for irrelevant say what the item contains instead.

Rules:
- Use only facts present in the input. Never invent content, numbers or names.
- Where it helps, quote a short phrase (at most 8 words) from the item's own title or description in double quotes, exactly as written there.
- Do not mention filters, filter checks, years, countries or coverage; those are shown separately.
- Keep item titles, database names and codes as they appear in the input. Plain prose, no markdown, no line breaks.

Return only a JSON object: {"sentence": "..."}"""


def quotes_supported(sentence: str, source: str) -> bool:
    """Every double-quoted phrase must occur in the input text (case-insensitive)."""
    src = re.sub(r"\s+", " ", source).lower()
    return all(re.sub(r"\s+", " ", q).lower().strip() in src for q in re.findall(r'"([^"]{2,})"', sentence))


def valid(sentence: str, source: str, max_words: int = 40) -> bool:
    s = sentence.strip()
    return bool(s) and "\n" not in s and len(s.split()) <= max_words and quotes_supported(s, source)


def write_sentence(base: str, text: str, verdict: str) -> str | None:
    body = {"model": MODEL, "max_tokens": 150, "temperature": 0.3,
            "messages": [{"role": "system", "content": SYSTEM},
                         {"role": "user", "content": f"{strip_similarity(text)}\nVerdict: {verdict}"}],
            "chat_template_kwargs": {"enable_thinking": False}}
    try:
        s = parse_json(post(base, body)["choices"][0]["message"]["content"])["sentence"].strip()
    except Exception:
        return None
    return s if valid(s, text) else None


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--out", required=True, help="directory for the one-sentence training set")
    ap.add_argument("--old-out", required=True, help="directory for the same inputs with the original narratives")
    ap.add_argument("--n", type=int, default=10000)
    ap.add_argument("--workers", type=int, default=192)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()

    data, out, old = Path(a.data), Path(a.out), Path(a.old_out)
    rows = [json.loads(l) for l in (data / "train.jsonl").read_text().splitlines()]
    idx = random.Random(a.seed).sample(range(len(rows)), min(a.n, len(rows)))
    cache = out / "sentences.jsonl"
    out.mkdir(parents=True, exist_ok=True)
    old.mkdir(parents=True, exist_ok=True)
    done = {d["i"]: d["s"] for d in map(json.loads, cache.read_text().splitlines())} if cache.exists() else {}
    todo = [i for i in idx if i not in done]
    servers = urls()

    def one(k: int):
        i = todo[k]
        return i, write_sentence(servers[k % len(servers)], rows[i]["input"], rows[i]["verdict"])

    with ThreadPoolExecutor(a.workers) as ex, cache.open("a") as f:
        for i, s in ex.map(one, range(len(todo))):
            if s:
                done[i] = s
                f.write(json.dumps({"i": i, "s": s}, ensure_ascii=False) + "\n")
    kept = [i for i in idx if i in done]
    with (out / "train.jsonl").open("w") as f, (old / "train.jsonl").open("w") as g:
        for i in kept:
            r = rows[i]
            f.write(json.dumps({**r, "output": f"{r['verdict']}\n{done[i]}"}, ensure_ascii=False) + "\n")
            g.write(json.dumps(r, ensure_ascii=False) + "\n")
    for d in (out, old):
        for split in ("val", "test"):
            link = d / f"{split}.jsonl"
            if not link.exists():
                os.symlink(os.path.relpath(data / f"{split}.jsonl", d), link)
    words = sorted(len(done[i].split()) for i in kept)
    print(f"{len(kept)} of {len(idx)} inputs have a valid sentence; median {words[len(words) // 2]} words, max {words[-1]}")


if __name__ == "__main__":
    main()
