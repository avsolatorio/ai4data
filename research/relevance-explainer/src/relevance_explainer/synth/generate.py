"""Synthetic training-data pipeline.

Stages (each writes JSONL and can be resumed):
  queries    teacher LLM writes user queries with filters for seed themes
  retrieve   run each query through Data360 search and sample results at varied ranks
  narrate    teacher LLM writes a verdict and short narrative for each (query, filters, result)

Example:
  python -m relevance_explainer.synth.generate queries --out data/pilot --seeds 12 --per-seed 3
  python -m relevance_explainer.synth.generate retrieve --out data/pilot
  python -m relevance_explainer.synth.generate narrate  --out data/pilot
"""
from __future__ import annotations

import argparse
import json
import os
import random
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from . import data360, llm

QUERY_MODEL = os.environ.get("QUERY_MODEL", "anthropic/claude-haiku-4.5")
NARRATE_MODEL = os.environ.get("NARRATE_MODEL", "anthropic/claude-sonnet-5.5")

# Query languages and sampling weights. Catalog metadata stays in English.
LANGUAGES = {"English": 40, "French": 12, "Spanish": 12, "Portuguese": 6, "Arabic": 8,
             "Chinese": 6, "Russian": 4, "Indonesian": 4, "Hindi": 4, "Turkish": 2, "Swahili": 2}

SEEDS = [
    "female labor force participation", "youth unemployment", "informal employment", "wages and earnings",
    "poverty headcount ratio", "income inequality and Gini", "shared prosperity", "household consumption",
    "primary school enrollment", "learning outcomes and test scores", "tertiary education", "teacher supply",
    "child stunting and malnutrition", "maternal mortality", "immunization coverage", "health expenditure",
    "access to clean water and sanitation", "access to electricity", "renewable energy share", "CO2 and greenhouse gas emissions",
    "forest area and deforestation", "agricultural productivity", "food prices and food security", "land and tenure",
    "GDP growth", "inflation and consumer prices", "public debt", "government revenue and taxes",
    "foreign direct investment", "merchandise trade", "remittances", "migration and refugees",
    "internet and mobile phone use", "financial inclusion and bank accounts", "digital payments", "business environment and firm surveys",
    "urban population and slums", "transport and roads", "housing affordability", "natural disasters and climate vulnerability",
    "governance and corruption", "gender-based violence", "women's legal rights", "social protection coverage",
    "population and fertility", "life expectancy and mortality", "disability", "water resources and withdrawals",
    "manufacturing value added", "tourism arrivals", "research and development spending", "air pollution",
]

QUERY_SYSTEM = """You write realistic search queries that analysts, journalists, students and policy staff type into a development data catalog (World Bank Data360 style), together with the filters they might apply."""

QUERY_USER = """Theme: "{seed}"

Write {n} distinct user searches related to this theme. Vary the style across them: short keywords, a full natural-language question, analyst jargon, and a vague layperson phrasing. Some searches should be narrower than the theme or touch a neighbouring topic.

The searches must be written in these languages, one per search, in this order: {languages}. Write each as a native speaker would type it, including local spellings of place names.

Each search has:
- "language": the language it is written in
- "query": the text typed, in that language
- "query_en": the same search in English (identical to "query" when it is already English)
- "filters": an object, possibly empty. Allowed keys: "countries" (list of country names), "years" (string like "2015-2022" or "2019"), "frequency" ("Annual", "Quarterly" or "Monthly"), "sex" ("female" or "male"), "age_group" (string), "area" ("urban" or "rural"). About 40% of searches should have no filters; the others should have one to three. Choose filters a real user would plausibly set, and include some that a typical indicator would probably not satisfy (a single small country, a recent single year, a quarterly frequency).

Return only a JSON array of objects."""

NARRATE_SYSTEM = """You judge whether a data catalog result answers a user's search, and explain the judgment in a short narrative that is shown next to the result.

Verdict definitions:
- relevant: the result measures what the query asks for and satisfies the filters that can be checked.
- partly_relevant: it measures a closely related concept, covers only part of what was asked (population group, geography, period, frequency), or fails a filter while being the right topic.
- irrelevant: it measures something different from what the query asks for.

Narrative rules:
- Write the narrative in the language named on the "Respond in:" line. Keep indicator names, database names and codes as they appear in the input.
- Two or three sentences, at most 60 words (or the equivalent length in that language).
- Use only facts present in the input. Never invent coverage, definitions or methods.
- Name the specific evidence: what the indicator measures, its population or unit, and the filter checks that pass or fail.
- For partly_relevant, say exactly which part of the query is satisfied and which is not.
- For irrelevant, say what the result measures and why that does not answer the query.
- Add insight where the metadata supports it, for example a definitional difference between related concepts or a usable workaround such as a related coverage window. Do not pad with generic statements.
- Do not restate the whole query. Do not use the words "query", "result" or "indicator" as the sentence subject more than once. No hedging filler.
- Plain prose, no bullet points, no markdown.

Return only a JSON object: {"verdict": "relevant" | "partly_relevant" | "irrelevant", "narrative": "..."}"""


def _read(p: Path) -> list[dict]:
    return [json.loads(l) for l in p.read_text().splitlines()] if p.exists() else []


def _append(p: Path, rows: list[dict]) -> None:
    with p.open("a") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def cmd_queries(a) -> None:
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    path = out / "queries.jsonl"
    done = {r["seed"] for r in _read(path)}
    seeds = [s for s in random.Random(a.seed).sample(SEEDS, min(a.seeds, len(SEEDS))) if s not in done]

    def one(seed: str) -> list[dict]:
        rng = random.Random(f"{a.seed}-{seed}")
        langs = rng.choices(list(LANGUAGES), weights=list(LANGUAGES.values()), k=a.per_seed)
        text, _ = llm.chat(QUERY_MODEL, QUERY_SYSTEM,
                           QUERY_USER.format(seed=seed, n=a.per_seed, languages=", ".join(langs)),
                           temperature=0.9, max_tokens=2500)
        return [{"seed": seed, "query": q["query"], "query_en": q.get("query_en") or q["query"],
                 "language": q.get("language") or "English", "filters": q.get("filters") or {}}
                for q in llm.parse_json(text) if q.get("query")]

    with ThreadPoolExecutor(a.workers) as ex:
        for rows in ex.map(one, seeds):
            _append(path, rows)
    print(f"queries: {len(_read(path))} total")


def cmd_retrieve(a) -> None:
    out = Path(a.out)
    rng = random.Random(a.seed)
    path = out / "pairs.jsonl"
    done = {r["query"] for r in _read(path)}
    todo = [q for q in _read(out / "queries.jsonl") if q["query"] not in done]

    def one(q: dict) -> list[dict]:
        hits = [(i + 1, data360.compact(h)) for i, h in enumerate(data360.search(q["query_en"], top=30))]
        hits = [(rank, r) for rank, r in hits if r]
        if not hits:
            return []
        top = hits[0]
        mid = rng.sample([h for h in hits[1:8]], min(a.mid, len(hits[1:8])))
        far = rng.sample([h for h in hits[8:]], min(a.far, len(hits[8:])))
        return [{**q, "rank": rank, "record": rec, "input": data360.render_input(q["query"], q["filters"], rec, q["language"])}
                for rank, rec in [top, *mid, *far]]

    with ThreadPoolExecutor(a.workers) as ex:
        for rows in ex.map(one, todo):
            _append(path, rows)
    print(f"pairs: {len(_read(path))} total")


def cmd_narrate(a) -> None:
    out = Path(a.out)
    path = out / "examples.jsonl"
    done = {(r["query"], r["record"]["id"]) for r in _read(path)}
    todo = [p for p in _read(out / "pairs.jsonl") if (p["query"], p["record"]["id"]) not in done]
    if a.limit:
        todo = todo[: a.limit]

    def one(p: dict) -> dict | None:
        try:
            text, usage = llm.chat(NARRATE_MODEL, NARRATE_SYSTEM, p["input"], temperature=0.4, max_tokens=400)
            j = llm.parse_json(text)
            if j["verdict"] not in {"relevant", "partly_relevant", "irrelevant"}:
                return None
            return {**p, "verdict": j["verdict"], "narrative": j["narrative"].strip(),
                    "teacher": NARRATE_MODEL, "usage": usage}
        except Exception as e:  # keep the run going; failures are retried on resume
            print("skip:", p["record"]["id"], type(e).__name__, e)
            return None

    n = 0
    with ThreadPoolExecutor(a.workers) as ex:
        for row in ex.map(one, todo):
            if row:
                _append(path, [row])
                n += 1
    print(f"examples: +{n}, {len(_read(path))} total")


def main() -> None:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    for name, fn in [("queries", cmd_queries), ("retrieve", cmd_retrieve), ("narrate", cmd_narrate)]:
        p = sub.add_parser(name)
        p.add_argument("--out", required=True)
        p.add_argument("--workers", type=int, default=8)
        p.add_argument("--seed", type=int, default=0)
        p.set_defaults(fn=fn)
        if name == "queries":
            p.add_argument("--seeds", type=int, default=len(SEEDS))
            p.add_argument("--per-seed", type=int, default=5)
        if name == "retrieve":
            p.add_argument("--mid", type=int, default=2)
            p.add_argument("--far", type=int, default=1)
        if name == "narrate":
            p.add_argument("--limit", type=int, default=0)
    a = ap.parse_args()
    a.fn(a)


if __name__ == "__main__":
    main()
