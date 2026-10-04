"""Build a fresh evaluation set whose searches and catalog items never occur in any data round.

Runs the normal generation stages against new themes, keeps only (query, item) pairs whose item id and title appear
in none of the round files (train, validation or test of every round), narrates those with the teacher, adds the
retrieval similarity, and records the teacher's thinking-mode verdict and a second model's verdict for each example.
The result is written in the viewer's example format.

  LLM_BASE_URL=http://localhost:8000/v1,http://localhost:8001/v1 QUERY_MODEL=qwen3.8-27b NARRATE_MODEL=qwen3.8-27b \
  LLM_EXTRA='{"chat_template_kwargs":{"enable_thinking":false}}' \
  python -m relevance_explainer.synth.fresh --out data/fresh --n 100
"""
from __future__ import annotations

import argparse
import json
import random
import re
from argparse import Namespace
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from . import generate as G
from . import llm

ROUNDS = [f"data/v{i}" for i in range(1, 8)]


def seen_items() -> tuple[set, set, set]:
    """Record ids, titles and themes that occur in any round (pairs or narrated examples)."""
    ids, titles, themes = set(), set(), set()
    for d in ROUNDS:
        for name in ("pairs.jsonl", "examples.jsonl", "queries.jsonl"):
            p = Path(d) / name
            if not p.exists():
                continue
            for line in p.read_text().splitlines():
                r = json.loads(line)
                themes.add(str(r.get("seed", "")).lower())
                rec = r.get("record") or r.get("anchor")
                if rec:
                    ids.add(str(rec.get("id")))
                    titles.add(str(rec.get("name", "")).strip().lower())
    return ids, titles, themes


def anchor_unseen(themes: list[str], ids: set, titles: set, a) -> list[dict]:
    """For (theme, source) draws, scan the top 30 results for items in no round, and have the teacher write searches
    that such an item answers fully or partly (the anchored-query prompt of the data rounds)."""
    rng = random.Random(a.seed + 7)
    combos = [(rng.choice(themes), rng.choices(list(G.SOURCE_WEIGHTS), weights=list(G.SOURCE_WEIGHTS.values()))[0])
              for _ in range(a.anchor_unseen)]

    def one(c):
        theme, kind = c
        src = G.SOURCES[kind]
        r = random.Random(f"au-{a.seed}-{c}")
        found = []
        for hit in src.search(theme, 30)[3:]:  # skip the very top, which earlier rounds covered most
            rec = src.hydrate(hit, theme)
            if rec and str(rec["id"]) not in ids and rec.get("name", "").strip().lower() not in titles and G._informative(rec, r, 0.0):
                found.append(rec)
                if len(found) >= 2:
                    break
        pairs = []
        for rec in found:
            item = "\n".join(src.render(rec))
            text, _ = llm.chat(G.QUERY_MODEL, G.QUERY_SYSTEM, G.ANCHOR_USER.format(label=src.label, item=item, languages=", ".join(G._langs(r, 2)),
                                                                                  filter_spec=G.FILTER_SPEC), temperature=0.8, max_tokens=1500)
            for q in G._items(text):
                if qq := G._clean(q, seed=theme, source=kind, intent=q.get("intent", "full")):
                    pairs.append({**qq, "rank": 0, "record": rec, "is_anchor": True,
                                  "input": G.render_input(qq["query"], qq["filters"], rec, qq["language"])})
        return pairs

    with ThreadPoolExecutor(a.retrieve_workers) as ex:
        return [p for ps in ex.map(lambda c: G._safe(one, c), combos) for p in ps]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="data/fresh")
    ap.add_argument("--n", type=int, default=100)
    ap.add_argument("--themes-per-area", type=int, default=8)
    ap.add_argument("--combos", type=int, default=90)
    ap.add_argument("--seed", type=int, default=4242)
    ap.add_argument("--workers", type=int, default=64)
    ap.add_argument("--retrieve-workers", type=int, default=12)
    ap.add_argument("--second-url", default="http://localhost:8002/v1")
    ap.add_argument("--second-model", default="gemma-4-31b")
    ap.add_argument("--teacher-urls", default="http://localhost:8000/v1,http://localhost:8001/v1")
    ap.add_argument("--anchor-unseen", type=int, default=0, help="(theme, source) draws for searches written for unseen items")
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)

    old_themes = set(t.lower() for t in G._seeds())
    ids, titles, round_themes = seen_items()
    print(f"seen in rounds: {len(ids)} item ids, {len(titles)} titles, {len(round_themes)} themes", flush=True)

    # 1. fresh themes only
    G.SEEDS = []
    G.THEMES_FILE = out / "themes_raw.json"
    if not G.THEMES_FILE.exists():
        G.cmd_themes(Namespace(per_area=a.themes_per_area, workers=8))
    raw = json.loads(G.THEMES_FILE.read_text())
    fresh = [t for t in raw if t.lower() not in old_themes and t.lower() not in round_themes]
    G.THEMES_FILE = out / "themes.json"
    G.THEMES_FILE.write_text(json.dumps(fresh, ensure_ascii=False, indent=0))
    print(f"themes: {len(raw)} generated, {len(fresh)} new", flush=True)

    # 2-3. queries (theme and item-anchored) and retrieval, with the usual stage code
    if not (out / "queries.jsonl").exists():
        G.cmd_queries(Namespace(out=str(out), workers=a.workers, seed=a.seed, combos=a.combos, per_combo=2))
        G.cmd_anchored(Namespace(out=str(out), workers=a.workers, seed=a.seed + 1, combos=a.combos // 2, items=2))
    if not (out / "pairs.jsonl").exists():
        G.cmd_retrieve(Namespace(out=str(out), workers=a.retrieve_workers, seed=a.seed, top=3, mid=1, far=2))

    # 4. keep only items that occur in no round
    pairs = G._read(out / "pairs.jsonl")
    unseen = [p for p in pairs if str(p["record"]["id"]) not in ids and p["record"].get("name", "").strip().lower() not in titles]
    print(f"pairs: {len(pairs)}; with an unseen item: {len(unseen)}", flush=True)
    keep = out / "pairs_unseen.jsonl"
    keep.write_text("".join(json.dumps(p, ensure_ascii=False) + "\n" for p in unseen))

    # 4b. searches written for unseen items found deep in the result lists (gives relevant and partly relevant cases)
    if a.anchor_unseen and not (out / "pairs_anchor_unseen.jsonl").exists():
        extra = anchor_unseen(fresh, ids, titles, a)
        (out / "pairs_anchor_unseen.jsonl").write_text("".join(json.dumps(p, ensure_ascii=False) + "\n" for p in extra))
        print(f"anchored pairs on unseen items: {len(extra)}", flush=True)
    if (out / "pairs_anchor_unseen.jsonl").exists():
        keep.write_text(keep.read_text() + (out / "pairs_anchor_unseen.jsonl").read_text())

    # 5. narrate the unseen pairs (pairs.jsonl is swapped for the stage, then restored; the stage resumes)
    full = (out / "pairs.jsonl").read_text()
    (out / "pairs.jsonl").write_text(keep.read_text())
    try:
        G.cmd_narrate(Namespace(out=str(out), workers=a.workers, seed=a.seed, limit=0))
    finally:
        (out / "pairs.jsonl").write_text(full)
    rows = G._read(out / "examples.jsonl")

    # 6. retrieval similarity, as for the training inputs
    from .signals import add_similarity_line, item_text, similarities
    sims = similarities([r["query"] for r in rows], [item_text(r["record"]) for r in rows])
    for r, s in zip(rows, sims):
        r["input"] = add_similarity_line(r["input"], float(s))

    # select a spread over verdicts, languages and sources
    rng = random.Random(a.seed)
    rng.shuffle(rows)
    want = {"relevant": int(a.n * 0.25), "partly_relevant": int(a.n * 0.35)}
    want["irrelevant"] = a.n - sum(want.values())
    pick, by_v, by_l, by_s, seen_q = [], Counter(), Counter(), Counter(), Counter()
    for strict in (True, False):
        for r in rows:
            if len(pick) >= a.n or r in pick or by_v[r["verdict"]] >= want[r["verdict"]]:
                continue
            if strict and (by_l[r["language"]] >= a.n // 8 or by_s[r["source"]] >= a.n * 0.4 or seen_q[r["query"]] >= 1):
                continue
            pick.append(r)
            by_v[r["verdict"]] += 1
            by_l[r["language"]] += 1
            by_s[r["source"]] += 1
            seen_q[r["query"]] += 1

    # 7. labels: teacher thinking mode and a second model
    from ..distill.audit import own_verdict
    from ..distill.teacher_checks import think_one
    teachers = a.teacher_urls.split(",")
    with ThreadPoolExecutor(32) as ex:
        think = list(ex.map(lambda k: think_one(teachers[k % len(teachers)], pick[k]["input"]), range(len(pick))))
    llm.BASE_URLS[:] = [a.second_url]
    with ThreadPoolExecutor(16) as ex:
        second = list(ex.map(lambda r: own_verdict(a.second_model, r["input"]), pick))

    examples = []
    for k, (r, t, s) in enumerate(zip(pick, think, second)):
        examples.append({"id": f"new-{k + 1}", "language": r["language"], "source": r["source"], "input": r["input"],
                         "reference": f"{r['verdict']}\n{r['narrative']}",
                         "labels": {"stored": r["verdict"], "thinking": t or r["verdict"], "gemma4": s or r["verdict"]},
                         "labels_missing": [n for n, v in (("thinking", t), ("gemma4", s)) if not v],
                         "offline": {"text": "", "judged": {}}, "unseen": True, "theme": r.get("seed")})
    examples.sort(key=lambda x: (x["language"] != "English", x["language"], x["source"]))
    (out / "examples_fresh.json").write_text(json.dumps(examples, ensure_ascii=False, indent=1))
    print(f"wrote {len(examples)} fresh examples; verdicts {dict(by_v)}; languages {dict(by_l)}; sources {dict(by_s)}")
    print("label agreement with thinking", sum(e["labels"]["stored"] == e["labels"]["thinking"] for e in examples),
          "with second model", sum(e["labels"]["stored"] == e["labels"]["gemma4"] for e in examples), "of", len(examples))


if __name__ == "__main__":
    main()
