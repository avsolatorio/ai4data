"""Per-language search aliases for countries and groupings (Gemma 4), merged into data/geo/lexicon.json.

For every country and grouping, the names, short forms, abbreviations and inflected forms that speakers of each of
the 11 languages type in searches, for example Russian case forms (в Кении), Arabic forms with and without the
article (تشيلي, الشيلي), Hindi transliteration variants, French and Spanish names of World Bank regions.

  LLM_BASE_URL=http://localhost:8002/v1 python -m relevance_explainer.synth.aliases
"""
from __future__ import annotations

import json
import os
import re
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from . import llm
from .geo import LANG_CODES
from .lexicon import _dedupe

LEX = Path("data/geo/lexicon.json")
GEO = Path("data/geo/geo.json")
MODEL = os.environ.get("LEXICON_MODEL", "gemma-4-31b")
SYSTEM = "You are a careful multilingual lexicographer who knows how people write place names in web searches."
PROMPT = """For each item below, list the names that {language} speakers type in searches for it: the standard {language} name, short and informal names, common abbreviations, spelling and transliteration variants, and {inflect}. 2 to 8 names each, lowercase, no explanations.
{items}

Return only a JSON object mapping each code exactly as given to a list of names."""
INFLECT = {
    "Russian": "the genitive, dative and prepositional case forms (for example кения, кении, кению)",
    "Arabic": "forms with and without the article al- and common spelling variants",
    "Turkish": "forms with common case suffixes after an apostrophe (for example kenya'da, kenya'nın)",
    "Hindi": "Devanagari spelling variants and the English name if commonly typed",
    "Chinese": "simplified short forms and common abbreviations",
    "Indonesian": "Indonesian and Malay spellings",
    "Swahili": "Swahili and English names as commonly typed",
}


def _parse(text: str):
    try:
        return llm.parse_json(text)
    except Exception:
        return llm.parse_json(re.sub(r",\s*([}\]])", r"\1", text))


def one(language: str, items: list[tuple[str, str]]) -> dict:
    text, _ = llm.chat(MODEL, SYSTEM, PROMPT.format(language=language, inflect=INFLECT.get(language, "common variants"),
                                                    items="\n".join(f"- {code}: {name}" for code, name in items)),
                       temperature=0.2, max_tokens=1800)
    return {k: _dedupe(v) for k, v in _parse(text).items() if isinstance(v, list)}


def main() -> None:
    geo = json.loads(GEO.read_text())
    lex = json.loads(LEX.read_text())
    countries = [(k, v["name"]) for k, v in geo["countries"].items()]
    groups = [(k, v["name"]) for k, v in geo["groups"].items()]
    jobs = []
    for lang in LANG_CODES:
        for kind, items, size in (("country_aliases", countries, 28), ("group_aliases", groups, 15)):
            if lex[lang].get(kind):
                continue
            for i in range(0, len(items), size):
                jobs.append((lang, kind, items[i : i + size]))

    def run(job):
        lang, kind, items = job
        for _ in range(3):
            try:
                return job, one(lang, items)
            except Exception as e:
                print("retry", lang, kind, type(e).__name__, str(e)[:60], flush=True)
        return job, {}

    with ThreadPoolExecutor(12) as ex:
        for (lang, kind, items), res in ex.map(run, jobs):
            codes = {c for c, _ in items}
            lex[lang].setdefault(kind, {}).update({k: v for k, v in res.items() if k in codes})
    LEX.write_text(json.dumps(lex, ensure_ascii=False, indent=1))
    for lang in LANG_CODES:
        print(lang, "countries", len(lex[lang].get("country_aliases", {})), "groups", len(lex[lang].get("group_aliases", {})))


if __name__ == "__main__":
    main()
