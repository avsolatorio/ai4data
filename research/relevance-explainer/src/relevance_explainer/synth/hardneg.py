"""Lexical hard negatives: real training searches paired with catalog records that share many of their words.

The challenge suite found the student calling an item relevant when it shares words with the search but is about
something else ("chemical stability" for a political stability index). The look-alike training searches were written by
an LLM told to invent such cases; these pairs come from the catalog instead. For each search (from a training-split
topic), records are ranked by TF-IDF cosine with the search's English gloss; among the lexical top 20, the record with the
lowest embedding similarity to the search is kept (many shared words, different meaning), and for a share of searches the
top lexical record is kept as it is (which is often relevant), so the labels go both ways. The rows are written as
challenge searches (phenomenon "lexical_neighbor") for `challenge label` and `challenge examples`.

  python -m relevance_explainer.synth.hardneg --out data/real_hardneg/h1 --n 8000
"""
from __future__ import annotations

import argparse
import json
import random
from collections import Counter
from pathlib import Path

import numpy as np

from . import generate as G


def load_searches(seed: int, n: int) -> list[dict]:
    rows = []
    for f in sorted(Path("data/real").glob("c*/examples.jsonl")):
        for line in f.open():
            r = json.loads(line)
            if G.split_of(r["topic"]) == "train" and r.get("search_en"):
                rows.append({"query": r["query"], "search_en": r["search_en"], "language": r["language"], "own": r["record"]["id"]})
    rng = random.Random(seed)
    rng.shuffle(rows)
    seen, out = set(), []
    for r in rows:
        if r["query"] not in seen:
            seen.add(r["query"])
            out.append(r)
        if len(out) >= n * 3:
            break
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="data/real_hardneg/h1")
    ap.add_argument("--n", type=int, default=8000)
    ap.add_argument("--top-share", type=float, default=0.25, help="share of searches paired with the top lexical record")
    ap.add_argument("--min-tfidf", type=float, default=0.3)
    ap.add_argument("--exclude", default="data/challenge/searches.jsonl", help="records of the held-out challenge suite")
    ap.add_argument("--seed", type=int, default=31)
    a = ap.parse_args()
    from sklearn.feature_extraction.text import TfidfVectorizer

    from .signals import item_text, similarities

    rng = random.Random(a.seed)
    excl = {json.loads(l)["record"]["id"] for l in Path(a.exclude).read_text().splitlines()} if a.exclude else set()
    recs = {}
    for f in sorted(Path("data/real").glob("c*/examples.jsonl")):
        for line in f.open():
            rec = json.loads(line)["record"]
            if rec["id"] not in excl:
                recs[rec["id"]] = rec
    for p in Path("data/pools").glob("*.jsonl"):
        if not p.name.endswith(("_fr.jsonl", "_es.jsonl")):
            for line in p.open():
                rec = json.loads(line)
                if rec["id"] not in excl:
                    recs[rec["id"]] = rec
    recs = list(recs.values())
    texts = [item_text(r) for r in recs]
    print("records:", len(recs), dict(Counter(r["kind"] for r in recs)))
    vec = TfidfVectorizer(stop_words="english", sublinear_tf=True, token_pattern=r"(?u)\b[^\W\d_]{3,}\b",  # words only: years matched unrelated items
                           min_df=2, max_df=0.2, ngram_range=(1, 2))
    X = vec.fit_transform(texts)
    searches = load_searches(a.seed, a.n)
    Q = vec.transform([s["search_en"] for s in searches])
    cands = []
    for i in range(0, Q.shape[0], 2000):
        S = (Q[i:i + 2000] @ X.T).toarray()
        top = np.argsort(-S, axis=1)[:, :20]
        for j, row in enumerate(top):
            s = searches[i + j]
            ks = [k for k in row if S[j, k] >= a.min_tfidf and recs[k]["id"] != s["own"]]
            if ks:
                cands.append((s, ks, [float(S[j, k]) for k in ks]))
    print("searches with lexical neighbours:", len(cands), "of", len(searches))
    flat = [(c, k) for c in cands for k in c[1]]
    sims = similarities([c[0]["query"] for c, _ in flat], [texts[k] for _, k in flat])
    emb = {}
    for (c, k), v in zip(flat, sims):
        emb[(id(c), k)] = float(v)
    rows = []
    for c in cands:
        s, ks, ts = c
        top = rng.random() < a.top_share
        k = ks[0] if top else min(ks, key=lambda k: emb[(id(c), k)])
        rows.append({"id": f"hn-{len(rows)}", "phenomenon": "lexical_neighbor", "language": s["language"], "record": recs[k],
                     "instruction": "", "query": s["query"], "writer_expected": None,
                     "note": f"{'top lexical' if top else 'lexical neighbour, lowest embedding similarity'}; tfidf {ts[ks.index(k)]:.2f}, embedding {emb[(id(c), k)]:.2f}"})
        if len(rows) >= a.n:
            break
    Path(a.out).mkdir(parents=True, exist_ok=True)
    (Path(a.out) / "searches.jsonl").write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows))
    print("written:", len(rows), dict(Counter(r["record"]["kind"] for r in rows)), dict(Counter(r["language"] for r in rows).most_common(6)))


if __name__ == "__main__":
    main()
