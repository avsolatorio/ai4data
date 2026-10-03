"""Add a cross-encoder reranker score to rendered inputs.

A search stack that reranks its results already has a score from a cross-encoder that reads the query and the item
together. This adds it after the retrieval-similarity line:

  Reranker score: 0.83

The score is the reranker's sigmoid output for (query, title + description + passage). Like the retrieval
similarity it is computed outside the 20 MB model.

  python -m relevance_explainer.synth.rerankscore --src data/f6 --out data/f6rr
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from .fieldsignals import parts

MODEL = "BAAI/bge-reranker-v2-m3"


def add_line(text: str, score: float) -> str:
    lines = text.split("\n")
    i = next((k for k, l in enumerate(lines) if l.startswith("Retrieval similarity:")), None)
    if i is None:
        i = next((k for k, l in enumerate(lines) if l.startswith("Filters:")), 0)
    lines.insert(i + 1, f"Reranker score: {score:.2f}")
    return "\n".join(lines)


def pair(text: str) -> tuple[str, str]:
    q, t, d, p = parts(text)
    return q, f"{t}. {d} {p}"[:1200]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--splits", nargs="+", default=["train", "val", "test"])
    a = ap.parse_args()
    from sentence_transformers import CrossEncoder

    ce = CrossEncoder(MODEL, device="cuda", max_length=512)
    ce.model.half()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    for split in a.splits:
        rows = [json.loads(l) for l in Path(a.src, f"{split}.jsonl").read_text().splitlines()]
        scores = ce.predict([pair(r["input"]) for r in rows], batch_size=128, show_progress_bar=False)
        with (out / f"{split}.jsonl").open("w") as f:
            for r, s in zip(rows, scores):
                r = dict(r)
                r["input"] = add_line(r["input"], float(s))
                r["rerank"] = float(s)
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        print(split, len(rows), flush=True)


if __name__ == "__main__":
    main()
