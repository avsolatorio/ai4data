"""Concept inventory: the vocabulary around each indicator, and searches built from it.

The challenge suite showed the student misses specific concept mappings ("number of inhabitants" for total population,
"inflation" for a consumer price index). Random records cover that vocabulary thinly, so this works through indicators
systematically: the teacher lists, per indicator, how people name it and what it is not; the training writers turn each
phrase into a search; labels come from synth/challenge.py (teacher and Gemma 4, agreed rows kept).

  python -m relevance_explainer.synth.concepts inventory --out data/concepts --n 2000
  python -m relevance_explainer.synth.concepts searches  --out data/concepts
  python -m relevance_explainer.synth.challenge label    --out data/concepts
  python -m relevance_explainer.synth.challenge examples --out data/concepts
"""
from __future__ import annotations

import argparse
import json
import random
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from . import generate as G
from . import llm
from .challenge import LANG_W
from .realistic import language_ok, writer

TEACHER = ("qwen3.8-27b", ["http://localhost:8000/v1", "http://localhost:8001/v1"])
CATS = {  # category: (phrases used per indicator, what a search with it should be)
    "everyday": (2, "asks for exactly what the indicator measures, in everyday words"),
    "synonym": (1, "asks for exactly what the indicator measures, using a synonym"),
    "acronym": (1, "asks for it with an acronym or abbreviation people use"),
    "broader": (1, "asks for a broad concept for which this indicator is the standard measure"),
    "proxy_for": (1, "asks for a concept this indicator only approximates"),
    "different_threshold": (1, "asks for the same measure with a different threshold, age band or unit"),
    "related_different": (1, "asks for a related but different concept"),
    "lookalike": (1, "uses words from the indicator's name with a different meaning"),
}
INV_PROMPT = """An indicator from a development-data catalog:
Name: {name}
Definition: {definition}
Unit: {unit}

List how people search for it and what it is not. Use real usage; leave a list empty when nothing fits. All phrases in English, 1 to 6 words each.
- everyday: 3 ways a non-specialist would ask for exactly this (e.g. "how many people live there" for total population)
- synonym: 2 synonyms or alternative technical names for exactly this
- acronym: up to 2 acronyms or abbreviations people really use for it
- broader: up to 2 broad concepts for which this indicator is the standard measure (e.g. "inflation" for a consumer price index)
- proxy_for: up to 2 concepts it only approximates without measuring them (e.g. "average income" for GDP per capita)
- different_threshold: up to 2 variants of it with a different threshold, age band or unit (e.g. "$3 a day poverty" for a $2.15 line)
- related_different: 2 related but different concepts (a neighbouring indicator, a cause or a consequence)
- lookalike: up to 2 phrases sharing a word with the name but meaning something else (e.g. "river bank erosion" for bank deposits)

Return only a JSON object with these eight keys, each a list of strings."""
SEARCH_PROMPT = """Write ONE search a person would type into a development-data catalog, in {language}. The search {what}: "{phrase}".
Express that wording naturally in {language} (translate it as a native speaker would; keep acronyms).{extra}
Form: {style}.
Return only a JSON object: {{"query": "..."}}"""
STYLES = ["a few keywords", "a short phrase", "a natural question", "two or three keywords, all lowercase"]


def cmd_inventory(a) -> None:
    rng = random.Random(a.seed)
    exclude = {json.loads(l)["record"]["id"] for l in Path("data/challenge/searches.jsonl").read_text().splitlines()}
    recs = {}
    for f in sorted(Path("data/real").glob("c*/examples.jsonl")):
        for l in f.open():
            r = json.loads(l)["record"]
            if r["kind"] == "indicator" and r["id"] not in exclude and len(r.get("definition") or "") > 40:
                recs.setdefault(r["name"].lower(), r)
    pool = list(recs.values())
    rng.shuffle(pool)
    pool = pool[: a.n]

    def one(rec):
        t, _ = llm.chat(TEACHER[0], "You know development statistics and how people search for them.",
                        INV_PROMPT.format(name=rec["name"], definition=(rec.get("definition") or "")[:400], unit=rec.get("unit") or "n/a"),
                        temperature=0.3, max_tokens=700, base_urls=TEACHER[1])
        j = llm.parse_json(t)
        inv = {k: [str(x).strip() for x in j.get(k) or [] if str(x).strip()][:3] for k in CATS}
        return [{"record": rec, "inventory": inv}] if sum(len(v) for v in inv.values()) >= 6 else []

    with ThreadPoolExecutor(a.workers) as ex:
        rows = [x for xs in ex.map(lambda r: G._safe(one, r), pool) for x in xs]
    Path(a.out).mkdir(parents=True, exist_ok=True)
    (Path(a.out) / "inventory.jsonl").write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows))
    print(f"inventory: {len(rows)} of {len(pool)} indicators;", dict(Counter(k for r in rows for k, v in r["inventory"].items() for _ in v)))


def cmd_searches(a) -> None:
    rng = random.Random(a.seed + 1)
    inv = [json.loads(l) for l in (Path(a.out) / "inventory.jsonl").read_text().splitlines()]
    jobs = []
    for i, row in enumerate(inv):
        rec = row["record"]
        for cat, (k, what) in CATS.items():
            for j, phrase in enumerate(rng.sample(row["inventory"].get(cat, []), min(k, len(row["inventory"].get(cat, []))))):
                extra = ""
                cs = rec.get("countries") or []
                if cs and rng.random() < 0.5:
                    extra += f" Mention the place {rng.choice(cs[:60])}."
                if rec.get("year_start") and rec.get("year_end") and rng.random() < 0.3:
                    extra += f" Mention the year {rng.randint(rec['year_start'], rec['year_end'])}."
                lang = rng.choices(list(LANG_W), weights=list(LANG_W.values()))[0]
                jobs.append({"id": f"{cat}-{i}-{j}", "phenomenon": f"concept_{cat}", "language": lang, "record": rec, "phrase": phrase,
                             "prompt": SEARCH_PROMPT.format(language=lang, what=what, phrase=phrase, extra=extra, style=rng.choice(STYLES))})

    def one(jb):
        for _ in range(3):
            model, urls = writer(jb["id"])
            t, _ = llm.chat(model, G.QUERY_SYSTEM, jb["prompt"], temperature=0.8, max_tokens=200, base_urls=urls)
            q = str(llm.parse_json(t).get("query") or "").strip()
            if q and language_ok(q, jb["language"]):
                return [{k: v for k, v in {**jb, "query": q, "writer": model}.items() if k != "prompt"}]
        return []

    with ThreadPoolExecutor(a.workers) as ex:
        rows = [x for xs in ex.map(lambda j: G._safe(one, j), jobs) for x in xs]
    (Path(a.out) / "searches.jsonl").write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows))
    print("searches:", len(rows), "of", len(jobs), dict(Counter(r["phenomenon"] for r in rows)))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("stage", choices=["inventory", "searches"])
    ap.add_argument("--out", default="data/concepts")
    ap.add_argument("--n", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=31)
    ap.add_argument("--workers", type=int, default=48)
    a = ap.parse_args()
    {"inventory": cmd_inventory, "searches": cmd_searches}[a.stage](a)


if __name__ == "__main__":
    main()
