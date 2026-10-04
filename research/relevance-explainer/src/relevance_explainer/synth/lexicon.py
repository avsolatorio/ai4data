"""Multilingual lexicons for query understanding, written once by the teacher and stored in data/geo/lexicon.json.

Per language: words and phrases for relative time, frequency, population groups and data types, generic geographic
scopes, and the local names and abbreviations of the groupings in geo.json.

  LLM_BASE_URL=http://localhost:8000/v1 LLM_EXTRA='{"chat_template_kwargs":{"enable_thinking":false}}' \
  python -m relevance_explainer.synth.lexicon
"""
from __future__ import annotations

import json
import os
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from . import llm
from .geo import LANG_CODES, load

OUT = Path("data/geo/lexicon.json")
MODEL = os.environ.get("LEXICON_MODEL", "gemma-4-31b")  # stronger than the teacher in Swahili and other smaller languages

SCHEMA = {
    "time": {"latest": "most recent / latest / current data", "recent": "recent / last few years",
             "since": "words placed before a year meaning from that year on (since, from, after)",
             "before": "words placed before a year meaning up to it (before, until, up to, prior to)",
             "between": "connectors in a year range (to, through, and, between)",
             "last": "last / past, as in 'last 10 years'", "years_word": "the word for years (all forms)",
             "pre_covid": "before the COVID-19 pandemic", "post_covid": "after or since the pandemic"},
    "frequency": {"annual": "annual / yearly", "quarterly": "quarterly", "monthly": "monthly", "weekly": "weekly", "daily": "daily"},
    "group": {"female": "women, girls, female", "male": "men, boys, male", "children": "children, under-5, infants",
              "youth": "youth, adolescents, young people", "elderly": "older people, elderly, aged 65+",
              "urban": "urban, cities, towns", "rural": "rural, villages, countryside", "poorest": "poor, poorest, bottom quintile, low-income households",
              "displaced": "refugees, internally displaced, IDPs, migrants in camps", "disability": "people with disabilities"},
    "datatype": {"indicator": "indicator, time series, statistics", "microdata": "microdata, survey data, census, household survey, unit-level data",
                 "document": "paper, report, working paper, study, publication", "dataset": "dataset, data file, CSV, spreadsheet, database"},
    "scope": {"global": "global, worldwide, all countries", "developing": "developing countries, Global South, emerging economies"},
}

PROMPT = """Write lexicons in {language} for a multilingual search engine over development data catalogs. For each key, list the words and short phrases that people actually type in {language} searches, including informal forms, common abbreviations and forms without accents where people often drop them. Lowercase. 3 to 10 entries per key. Include an English term only if {language} speakers commonly type it in searches.

Keys and meanings:
{schema}

Return only a JSON object: {{"time": {{key: [..]}}, "frequency": {{..}}, "group": {{..}}, "datatype": {{..}}, "scope": {{..}}}}"""

GROUP_PROMPT = """Give the names that {language} speakers use in searches for each of these regions and country groupings, including common abbreviations (for example MENA, SSA, LAC, ASS, ALC, PMA, LDCs). 1 to 5 names each, lowercase.
{groups}

Return only a JSON object mapping each English name exactly as given to a list of {language} names."""

SYSTEM = "You are a careful multilingual lexicographer."


def one(language: str, groups: list[str]) -> dict:
    schema = "\n".join(f"- {sec}.{k}: {v}" for sec, d in SCHEMA.items() for k, v in d.items())
    text, _ = llm.chat(MODEL, SYSTEM, PROMPT.format(language=language, schema=schema), temperature=0.2, max_tokens=2000)
    res = {sec: {k: _dedupe(v) for k, v in d.items() if isinstance(v, list)} for sec, d in _parse(text).items() if isinstance(d, dict)}
    res["groupings"] = {}
    for i in range(0, len(groups), 12):
        chunk = groups[i : i + 12]
        text, _ = llm.chat(MODEL, SYSTEM, GROUP_PROMPT.format(language=language, groups="\n".join(f"- {g}" for g in chunk)),
                           temperature=0.2, max_tokens=1500)
        res["groupings"].update({k: _dedupe(v) for k, v in _parse(text).items() if k in chunk and isinstance(v, list)})
    return res


def _parse(text: str):
    """JSON from the model, tolerating trailing commas."""
    import re
    try:
        return llm.parse_json(text)
    except Exception:
        return llm.parse_json(re.sub(r",\s*([}\]])", r"\1", text))


def _dedupe(xs: list) -> list[str]:
    out = []
    for x in xs:
        x = str(x).strip().lower()
        if x and x not in out:
            out.append(x)
    return out


def main() -> None:
    geo = load()
    groups = sorted({g["name"] for g in geo["groups"].values()})
    out = json.loads(OUT.read_text()) if OUT.exists() else {}
    todo = [l for l in LANG_CODES if l not in out]
    with ThreadPoolExecutor(len(todo) or 1) as ex:
        for lang, res in zip(todo, ex.map(lambda l: _safe(l, groups), todo)):
            if res:
                out[lang] = res
                print(lang, {k: len(v) for k, v in res.items() if isinstance(v, dict)}, flush=True)
    OUT.write_text(json.dumps(out, ensure_ascii=False, indent=1))
    print(f"wrote {OUT}: {len(out)} languages")


def _safe(lang: str, groups: list[str]):
    for _ in range(3):
        try:
            return one(lang, groups)
        except Exception as e:  # malformed JSON: retry
            print("retry", lang, type(e).__name__, str(e)[:80], flush=True)
    return None


if __name__ == "__main__":
    main()
