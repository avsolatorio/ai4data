"""Variants of the realistic test set that differ only in the "Retrieval similarity" line, to measure how much the model
depends on the retriever's score: our retriever (as trained), other retrievers, the line removed, a constant, shuffled.

  python -m relevance_explainer.synth.sim_variants --out data/sim_variants
"""
from __future__ import annotations

import argparse
import json
import random
import re
from pathlib import Path

import numpy as np

from .signals import MODEL, item_text

RETRIEVERS = {"e5small": "ai4data/devdata-search-multilingual-e5-small-cmnrl",
              "mlminilm": "ai4data/devdata-search-paraphrase-ml-minilm-cmnrl",
              "minilm_en": "ai4data/devdata-search-minilm-l6-cmnrl"}


def cosine(model: str, queries: list[str], items: list[str]) -> np.ndarray:
    from sentence_transformers import SentenceTransformer
    m = SentenceTransformer(model, device="cuda")
    q = m.encode(queries, batch_size=128, normalize_embeddings=True, show_progress_bar=False)
    d = m.encode(items, batch_size=128, normalize_embeddings=True, show_progress_bar=False)
    return (q * d).sum(1)


def set_line(inp: str, sim: float | None) -> str:
    if sim is None:
        return re.sub(r"\nRetrieval similarity: [-0-9.]+", "", inp)
    return re.sub(r"Retrieval similarity: [-0-9.]+", f"Retrieval similarity: {sim:.2f}", inp)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--test", default="data/eval_real/itemfirst/test.jsonl")
    ap.add_argument("--src", default="data/real_test/examples_labeled.jsonl")
    ap.add_argument("--out", default="data/sim_variants")
    a = ap.parse_args()
    rows = [json.loads(l) for l in Path(a.test).read_text().splitlines()]
    src = [json.loads(l) for l in Path(a.src).read_text().splitlines()]
    assert len(rows) == len(src)
    for r, s in zip(rows, src):  # the eval set was written in the source order
        assert f"Search: {s['query']}\n" in r["input"]
    queries, items = [s["query"] for s in src], [item_text(s["record"]) for s in src]
    orig = np.array([s["similarity"] for s in src])
    sims = {"orig": orig, "ours_recomputed": cosine(MODEL, queries, items)}
    for k, m in RETRIEVERS.items():
        sims[k] = cosine(m, queries, items)
    rng = random.Random(0)
    shuffled = orig.copy()
    rng.shuffle(shuffled)
    sims["shuffled"], sims["const_median"] = shuffled, np.full(len(orig), float(np.median(orig)))
    stats = {k: {"mean": round(float(v.mean()), 3), "sd": round(float(v.std()), 3), "p10": round(float(np.percentile(v, 10)), 3),
                 "p90": round(float(np.percentile(v, 90)), 3), "corr_with_ours": round(float(np.corrcoef(v, orig)[0, 1]), 3) if v.std() > 0 else None}
             for k, v in sims.items()}
    sims["none"] = None
    for k, v in sims.items():
        d = Path(a.out) / k
        d.mkdir(parents=True, exist_ok=True)
        text = "".join(json.dumps({**r, "input": set_line(r["input"], None if v is None else float(v[i]))}, ensure_ascii=False) + "\n"
                       for i, r in enumerate(rows))
        for split in ("test", "val"):
            (d / f"{split}.jsonl").write_text(text)
        (d / "train.jsonl").write_text(text.split("\n", 1)[0] + "\n")
    (Path(a.out) / "stats.json").write_text(json.dumps(stats, indent=1))
    for k, s in stats.items():
        print(k, s)


if __name__ == "__main__":
    main()
