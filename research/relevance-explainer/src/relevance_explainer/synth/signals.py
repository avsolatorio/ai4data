"""Retrieval signals for each (query, item) pair: embedding similarity from the repo's multilingual search model.

The similarity is what a production retriever already has, so it can be given to the explainer as an input.
"""
from __future__ import annotations

import json
import re
from pathlib import Path

import numpy as np

MODEL = "ai4data/devdata-search-granite-97m-multilingual-cmnrl"


def item_text(rec: dict) -> str:
    """Title plus a short description, in the order a search index would embed them."""
    desc = rec.get("definition") or rec.get("abstract") or rec.get("notes") or ""
    return re.sub(r"\s+", " ", f"{rec['name']}. {desc}")[:600]


def similarities(queries: list[str], items: list[str], model_name: str = MODEL, batch: int = 256) -> np.ndarray:
    from sentence_transformers import SentenceTransformer

    # GPU first; CPU when the GPU is full (shared with vLLM and training: CUBLAS allocation failures on 2026-10-07)
    for device in ("cuda", "cpu"):
        try:
            m = SentenceTransformer(model_name, device=device)
            q = m.encode(queries, batch_size=batch, normalize_embeddings=True, show_progress_bar=False)
            d = m.encode(items, batch_size=batch, normalize_embeddings=True, show_progress_bar=False)
            return (q * d).sum(1)
        except RuntimeError as e:
            if device == "cpu":
                raise
            print(f"similarities: GPU failed ({str(e)[:80]}); using the CPU")


def add_similarity_line(text: str, sim: float) -> str:
    """Insert 'Retrieval similarity: 0.61' after the Filters line of a rendered input."""
    lines = text.split("\n")
    i = next((k for k, l in enumerate(lines) if l.startswith("Filters:")), 0)
    lines.insert(i + 1, f"Retrieval similarity: {sim:.2f}")
    return "\n".join(lines)


def load_examples(dirs: list[str]) -> list[dict]:
    seen, rows = set(), []
    for d in dirs:
        for line in (Path(d) / "examples.jsonl").read_text().splitlines():
            r = json.loads(line)
            key = (r["query"], r["record"]["id"])
            if key not in seen and "publication year" not in r["input"]:
                seen.add(key)
                rows.append(r)
    return rows
