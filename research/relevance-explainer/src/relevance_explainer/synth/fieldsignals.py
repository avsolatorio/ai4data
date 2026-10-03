"""Field-level similarity signals added to rendered inputs.

The retriever's similarity compares the query with title plus description as one text. This adds the similarity of
the query with each field separately, which separates "the topic matches" from "only the name matches":

  Field similarity: title 0.62; description 0.41; passage 0.55

The passage value appears only for documents. All values use the same multilingual search model as the retrieval
similarity.

  python -m relevance_explainer.synth.fieldsignals --src data/f6 --out data/f6sig
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import numpy as np

from .signals import MODEL

DESC_FIELDS = ("Definition", "Abstract", "Description")


def field(text: str, name: str) -> str:
    m = re.search(rf"^{name}: (.+)$", text, re.M)
    return m.group(1).strip() if m else ""


def parts(text: str) -> tuple[str, str, str, str]:
    desc = next((field(text, f) for f in DESC_FIELDS if field(text, f)), "")
    return field(text, "Query"), field(text, "Result"), desc[:500], field(text, "Retrieved passage")[:500]


def add_line(text: str, title: float, desc: float | None, passage: float | None) -> str:
    vals = [f"title {title:.2f}"] + ([f"description {desc:.2f}"] if desc is not None else []) + \
           ([f"passage {passage:.2f}"] if passage is not None else [])
    lines = text.split("\n")
    i = next((k for k, l in enumerate(lines) if l.startswith("Retrieval similarity:")), None)
    if i is None:
        i = next((k for k, l in enumerate(lines) if l.startswith("Filters:")), 0)
    lines.insert(i + 1, "Field similarity: " + "; ".join(vals))
    return "\n".join(lines)


def encode_unique(model, texts: list[str]) -> dict[str, np.ndarray]:
    uniq = sorted({t for t in texts if t})
    emb = model.encode(uniq, batch_size=512, normalize_embeddings=True, show_progress_bar=False)
    return dict(zip(uniq, emb))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--splits", nargs="+", default=["train", "val", "test"])
    a = ap.parse_args()
    from sentence_transformers import SentenceTransformer

    model = SentenceTransformer(MODEL, device="cuda")
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    for split in a.splits:
        rows = [json.loads(l) for l in Path(a.src, f"{split}.jsonl").read_text().splitlines()]
        ps = [parts(r["input"]) for r in rows]
        emb = encode_unique(model, [t for p in ps for t in p])
        with (out / f"{split}.jsonl").open("w") as f:
            for r, (q, t, d, p) in zip(rows, ps):
                sim = lambda x: float(emb[q] @ emb[x]) if q and x else None  # noqa: E731
                r = dict(r)
                r["input"] = add_line(r["input"], sim(t) or 0.0, sim(d), sim(p))
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        print(split, len(rows), "rows;", len(emb), "unique texts", flush=True)


if __name__ == "__main__":
    main()
