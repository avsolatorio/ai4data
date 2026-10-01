"""Synthetic training-data pipeline over several open catalogs.

Stages (each appends to a JSONL file and can be resumed):
  queries    teacher LLM writes multilingual searches with filters for (theme, source) pairs
  anchored   teacher LLM writes searches anchored to a real catalog item: one it answers
             fully and one it answers only partly
  retrieve   run each search against its source and sample results at varied ranks;
             anchored items are always included
  narrate    teacher LLM writes a verdict and short narrative for each pair

Example:
  python -m relevance_explainer.synth.generate queries  --out data/pilot --combos 24
  python -m relevance_explainer.synth.generate anchored --out data/pilot --combos 24
  python -m relevance_explainer.synth.generate retrieve --out data/pilot
  python -m relevance_explainer.synth.generate narrate  --out data/pilot
"""
from __future__ import annotations

import argparse
import json
import os
import random
import zlib
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from . import llm
from .sources import SOURCES, render_input

QUERY_MODEL = os.environ.get("QUERY_MODEL", "anthropic/claude-haiku-4.5")
NARRATE_MODEL = os.environ.get("NARRATE_MODEL", "anthropic/claude-sonnet-5.5")

# Query languages and sampling weights. Catalog metadata stays in English.
LANGUAGES = {"English": 40, "French": 12, "Spanish": 12, "Portuguese": 6, "Arabic": 8,
             "Chinese": 6, "Russian": 4, "Indonesian": 4, "Hindi": 4, "Turkish": 2, "Swahili": 2}
# Share of searches per catalog source.
SOURCE_WEIGHTS = {"indicator": 35, "document": 25, "study": 20, "dataset": 20}

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
    "internally displaced persons", "cash transfers and safety nets", "conflict and fragility", "flood and drought impacts",
    "cholera and disease outbreaks", "household survey on living standards", "labor force survey", "demographic and health survey",
    "school feeding and child labor", "climate adaptation finance", "informal sector firms", "COVID-19 impacts on households",
]

QUERY_SYSTEM = """You write realistic search queries that analysts, journalists, students and policy staff type into open development data catalogs, together with the filters they might apply."""

FILTER_SPEC = """- "filters": an object, possibly empty. Allowed keys: "countries" (list of country names in English), "years" (string like "2015-2022" or "2019"), "frequency" ("Annual", "Quarterly" or "Monthly"), "sex" ("female" or "male"), "age_group" (string), "area" ("urban" or "rural")."""

QUERY_USER = """Theme: "{seed}"
The user is searching a catalog of: {label}.

Write {n} distinct user searches related to this theme, suited to that kind of catalog. Vary the style across them: short keywords, a full natural-language question, analyst jargon, and a vague layperson phrasing. Some searches should be narrower than the theme or touch a neighbouring topic.

The searches must be written in these languages, one per search, in this order: {languages}. Write each as a native speaker would type it, including local spellings of place names.

Each search has:
- "language": the language it is written in
- "query": the text typed, in that language
- "query_en": the same search in English (identical to "query" when it is already English)
{filter_spec}
About 40% of searches should have no filters; the others should have one to three. Choose filters a real user would plausibly set, and include some that a typical item would probably not satisfy (a single small country, a recent single year, a quarterly frequency).

Return only a JSON array of objects."""

ANCHOR_USER = """Here is an item from an open data catalog ({label}):

{item}

Write two searches that a user might type, in these languages in this order: {languages}. Each search has:
- "intent": "full" for the first, "partial" for the second
- "language", "query" (in that language), "query_en" (English version)
{filter_spec}

Search 1 ("full"): this item is an excellent answer. Any filters must be ones the item satisfies, judging only from the item text.
Search 2 ("partial"): the item addresses the topic but answers only part of the need. Add a requirement the item does not meet: a different population group, a finer disaggregation, a country or period outside its coverage, or a different frequency.

Write natural searches. Do not copy the item title word for word. Return only a JSON array of two objects."""

NARRATE_SYSTEM = """You judge whether a data catalog item answers a user's search, and explain the judgment in a short narrative that is shown next to the item. Items can be statistical indicators, survey microdata studies, research documents with a retrieved passage, or humanitarian datasets.

Verdict definitions:
- relevant: the item addresses what the query asks for and satisfies the filters that can be checked.
- partly_relevant: it addresses a closely related concept, covers only part of what was asked (population group, geography, period, frequency, level of detail), or fails a filter while being the right topic.
- irrelevant: it addresses something different from what the query asks for.

Narrative rules:
- Write the narrative in the language named on the "Respond in:" line. Keep item titles, database names and codes as they appear in the input.
- Two or three sentences, at most 60 words (or the equivalent length in that language).
- Use only facts present in the input. Never invent coverage, definitions, methods, variables or findings.
- Trust the "Filter check" line: do not contradict it. A filter with no check (sex, age group, area) can be assessed only if the item text states it; otherwise say it cannot be confirmed.
- When the metadata is too sparse to show what the item contains, say so briefly and judge from the title alone. Do not infer content that is not written.
- Name the specific evidence: what the item measures or contains, its population, unit or scope, and the filter checks that pass or fail.
- For partly_relevant, say exactly which part of the query is satisfied and which is not.
- For irrelevant, say what the item contains and why that does not answer the query.
- For a document, say whether the retrieved passage itself addresses the query or only the paper's general topic. If no passage was retrieved, judge from the abstract.
- Add insight where the metadata supports it, for example a definitional difference between related concepts or a usable workaround such as a related coverage window. Do not pad with generic statements.
- Do not restate the whole query. Avoid opening with the words "This item" or "This result". No hedging filler.
- Plain prose, no bullet points, no markdown.

Return only a JSON object: {"verdict": "relevant" | "partly_relevant" | "irrelevant", "narrative": "..."}"""


def _read(p: Path) -> list[dict]:
    return [json.loads(l) for l in p.read_text().splitlines()] if p.exists() else []


def _append(p: Path, rows: list[dict]) -> None:
    with p.open("a") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def _combos(a, done: set) -> list[tuple[str, str]]:
    rng = random.Random(a.seed)
    out: list[tuple[str, str]] = []
    while len(out) < a.combos:
        c = (rng.choice(SEEDS), rng.choices(list(SOURCE_WEIGHTS), weights=list(SOURCE_WEIGHTS.values()))[0])
        if c not in out and c not in done:
            out.append(c)
    return out


def _langs(rng: random.Random, k: int) -> list[str]:
    return rng.choices(list(LANGUAGES), weights=list(LANGUAGES.values()), k=k)


_CODES = {"en": "English", "fr": "French", "es": "Spanish", "pt": "Portuguese", "ar": "Arabic", "zh": "Chinese",
          "ru": "Russian", "id": "Indonesian", "hi": "Hindi", "tr": "Turkish", "sw": "Swahili"}


def _language(name: str | None) -> str | None:
    """Map code or loose names to a canonical language name; None when unknown."""
    n = (name or "English").strip()
    n = _CODES.get(n.lower(), n.title())
    return n if n in LANGUAGES else None


def _safe(fn, arg) -> list[dict]:
    """Run one unit of work; report and skip failures so a stage can finish and be resumed."""
    try:
        return fn(arg)
    except Exception as e:
        print("skip:", arg if isinstance(arg, tuple) else "", type(e).__name__, e)
        return []


def _items(text: str) -> list[dict]:
    """Parse a teacher response into a list of objects, unwrapping {"searches": [...]} and skipping non-objects."""
    j = llm.parse_json(text)
    if isinstance(j, dict):
        j = next((v for v in j.values() if isinstance(v, list)), [j])
    return [x for x in j if isinstance(x, dict)]


def _clean(q: dict, **extra) -> dict | None:
    lang = _language(q.get("language"))
    if not q.get("query") or not lang or not isinstance(q.get("filters") or {}, dict):
        return None
    return {"query": q["query"], "query_en": q.get("query_en") or q["query"],
            "language": lang, "filters": q.get("filters") or {}, **extra}


def cmd_queries(a) -> None:
    path = Path(a.out) / "queries.jsonl"
    path.parent.mkdir(parents=True, exist_ok=True)
    rows0 = _read(path)
    todo = _combos(a, {(r["seed"], r["source"]) for r in rows0})

    def one(c: tuple[str, str]) -> list[dict]:
        seed, kind = c
        langs = _langs(random.Random(f"{a.seed}-{c}"), a.per_combo)
        text, _ = llm.chat(QUERY_MODEL, QUERY_SYSTEM,
                           QUERY_USER.format(seed=seed, label=SOURCES[kind].label, n=a.per_combo,
                                             languages=", ".join(langs), filter_spec=FILTER_SPEC),
                           temperature=0.9, max_tokens=2500)
        return [r for q in _items(text) if (r := _clean(q, seed=seed, source=kind, intent="theme"))]

    with ThreadPoolExecutor(a.workers) as ex:
        for rows in ex.map(lambda c: _safe(one, c), todo):
            _append(path, rows)
    print(f"queries: {len(_read(path))} total")


def cmd_anchored(a) -> None:
    path = Path(a.out) / "queries.jsonl"
    path.parent.mkdir(parents=True, exist_ok=True)
    todo = _combos(a, {(r["seed"], r["source"]) for r in _read(path) if r["intent"] != "theme"})
    todo = [c for c in todo]

    def one(c: tuple[str, str]) -> list[dict]:
        seed, kind = c
        rng = random.Random(f"anchor-{a.seed}-{c}")
        src = SOURCES[kind]
        hits = src.search(seed, 10)
        rng.shuffle(hits)
        rows: list[dict] = []
        for hit in hits[: a.items]:
            rec = src.hydrate(hit, seed)
            if not rec:
                continue
            item = "\n".join(src.render(rec))
            text, _ = llm.chat(QUERY_MODEL, QUERY_SYSTEM,
                               ANCHOR_USER.format(label=src.label, item=item, languages=", ".join(_langs(rng, 2)),
                                                  filter_spec=FILTER_SPEC),
                               temperature=0.8, max_tokens=1500)
            for q in _items(text):
                if r := _clean(q, seed=seed, source=kind, intent=q.get("intent", "full"), anchor=rec):
                    rows.append(r)
        return rows

    with ThreadPoolExecutor(a.workers) as ex:
        for rows in ex.map(lambda c: _safe(one, c), todo):
            _append(path, rows)
    print(f"queries: {len(_read(path))} total")


def _informative(rec: dict, rng: random.Random, keep_sparse: float = 0.15) -> bool:
    """Drop most records whose metadata is too thin to judge; keep a few so the model sees sparse input."""
    text = rec.get("abstract") or rec.get("definition") or rec.get("notes") or ""
    sparse = len(text) < 40 and not (rec.get("topics") or rec.get("tags"))
    return not sparse or rng.random() < keep_sparse


def cmd_retrieve(a) -> None:
    out = Path(a.out)
    rng = random.Random(a.seed)
    path = out / "pairs.jsonl"
    done = {(r["query"], r["record"]["id"]) for r in _read(path)}
    todo = _read(out / "queries.jsonl")
    seen_q = {r["query"] for r in _read(path)}
    todo = [q for q in todo if q["query"] not in seen_q]

    def one(q: dict) -> list[dict]:
        src = SOURCES[q["source"]]
        hits = list(enumerate(src.search(q["query_en"], 30), 1))
        top = a.top
        chosen = hits[:top] + rng.sample(hits[top:8], min(a.mid, len(hits[top:8]))) + rng.sample(hits[8:], min(a.far, len(hits[8:])))
        pairs = []
        for rank, hit in chosen:
            rec = src.hydrate(hit, q["query_en"])
            if rec and _informative(rec, rng):
                pairs.append((rank, rec))
        anchor = q.get("anchor")
        if anchor and anchor["id"] not in {r["id"] for _, r in pairs}:
            pairs.append((0, anchor))  # rank 0 marks a record that was not drawn from this search
        return [{**{k: v for k, v in q.items() if k != "anchor"}, "rank": rank, "record": rec,
                 "is_anchor": bool(anchor and rec["id"] == anchor["id"]),
                 "input": render_input(q["query"], q["filters"], rec, q["language"])}
                for rank, rec in pairs if (q["query"], rec["id"]) not in done]

    def safe(q: dict) -> list[dict]:
        try:
            return one(q)
        except Exception as e:
            print("skip query:", q["query_en"][:60], type(e).__name__, e)
            return []

    with ThreadPoolExecutor(a.workers) as ex:
        for rows in ex.map(safe, todo):
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


def split_of(seed_theme: str) -> str:
    """Assign a whole theme to one split so validation and test themes are unseen in training."""
    b = zlib.crc32(seed_theme.encode()) % 100
    return "train" if b < 80 else "val" if b < 90 else "test"


def balance(rows: list[dict], max_ratio: float, rng: random.Random) -> list[dict]:
    """Downsample so no verdict has more than max_ratio times the rarest verdict's count."""
    by: dict[str, list[dict]] = {}
    for r in rows:
        by.setdefault(r["verdict"], []).append(r)
    if len(by) < 2:
        return rows
    cap = int(min(len(v) for v in by.values()) * max_ratio)
    out = [r for v in by.values() for r in (rng.sample(v, cap) if len(v) > cap else v)]
    rng.shuffle(out)
    return out


def cmd_assemble(a) -> None:
    out = Path(a.out)
    rng = random.Random(a.seed)
    seen, rows = set(), []
    for r in _read(out / "examples.jsonl"):
        key = (r["query"], r["record"]["id"])
        if key not in seen:
            seen.add(key)
            rows.append(r)
    splits: dict[str, list[dict]] = {"train": [], "val": [], "test": []}
    for r in rows:
        splits[split_of(r["seed"])].append(r)
    for name, rs in splits.items():
        rs = balance(rs, a.max_ratio, rng) if name == "train" or a.balance_eval else rs
        with (out / f"{name}.jsonl").open("w") as f:
            for r in rs:
                f.write(json.dumps({
                    "input": r["input"], "output": f"{r['verdict']}\n{r['narrative']}",
                    "verdict": r["verdict"], "source": r["source"], "language": r["language"],
                    "seed": r["seed"], "intent": r["intent"], "is_anchor": r["is_anchor"],
                }, ensure_ascii=False) + "\n")
        c = {v: sum(r["verdict"] == v for r in rs) for v in ("relevant", "partly_relevant", "irrelevant")}
        print(f"{name}: {len(rs)} {c}")


def main() -> None:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    for name, fn in [("queries", cmd_queries), ("anchored", cmd_anchored), ("retrieve", cmd_retrieve),
                     ("narrate", cmd_narrate), ("assemble", cmd_assemble)]:
        p = sub.add_parser(name)
        p.add_argument("--out", required=True)
        p.add_argument("--workers", type=int, default=8)
        p.add_argument("--seed", type=int, default=0)
        p.set_defaults(fn=fn)
        if name in {"queries", "anchored"}:
            p.add_argument("--combos", type=int, default=20, help="number of (theme, source) pairs")
        if name == "queries":
            p.add_argument("--per-combo", type=int, default=4)
        if name == "anchored":
            p.add_argument("--items", type=int, default=2, help="catalog items per combo")
        if name == "retrieve":
            p.add_argument("--top", type=int, default=3, help="always take this many top-ranked results")
            p.add_argument("--mid", type=int, default=1, help="random extra results from the remaining top 8")
            p.add_argument("--far", type=int, default=1, help="random results from ranks 9-30")
        if name == "narrate":
            p.add_argument("--limit", type=int, default=0)
        if name == "assemble":
            p.add_argument("--max-ratio", type=float, default=1.5, help="largest verdict class / smallest, in train")
            p.add_argument("--balance-eval", action="store_true", help="also balance val and test")
    a = ap.parse_args()
    a.fn(a)


if __name__ == "__main__":
    main()
