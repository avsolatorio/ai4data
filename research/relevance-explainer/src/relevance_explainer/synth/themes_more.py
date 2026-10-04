"""Expand the search themes: development areas x angles, written by two models for variety.

Writes the merged list to --out (default data/themes.json, after copying the original to data/themes_orig.json if
that backup does not exist). Themes reserved for the unseen test set (data/fresh/themes.json) are excluded, and near
duplicates (same words ignoring case, accents and order) are dropped.

  python -m relevance_explainer.synth.themes_more --per 40
"""
from __future__ import annotations

import argparse
import json
import random
import re
import shutil
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from . import generate as G
from . import llm
from .understand import norm

ANGLES = [
    "how it is measured: indicators, rates, indices and their breakdowns",
    "household and enterprise survey topics and questionnaire modules",
    "policy programs, reforms and their evaluation",
    "humanitarian needs, crises and response operations",
    "infrastructure, services and access at local level",
    "people and groups: gender, age, disability, displacement, ethnicity",
    "markets, prices, trade flows and finance",
    "environment, risk and climate links",
    "public spending, institutions and administrative data",
    "specific everyday issues people search in plain words (not jargon)",
]
WRITERS = {"qwen3.8-27b": ["http://localhost:8000/v1", "http://localhost:8001/v1"], "gemma-4-31b": ["http://localhost:8002/v1"]}
PROMPT = """List {n} distinct, specific topics within "{area}", from this angle: {angle}.
They are topics people look for in open development data catalogs (statistical indicators, survey microdata, research papers, humanitarian datasets). Each is a short English noun phrase of 2 to 7 words. Vary the level of specificity; no two share their main keyword; no place names or years.
Return only a JSON array of strings."""


def key(t: str) -> str:
    return " ".join(sorted(re.findall(r"\w+", norm(t))))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--per", type=int, default=40)
    ap.add_argument("--out", default="data/themes.json")
    a = ap.parse_args()
    out = Path(a.out)
    backup = out.with_name("themes_orig.json")
    if out.exists() and not backup.exists():
        shutil.copy(out, backup)
    have = json.loads(out.read_text()) if out.exists() else []
    reserved = {key(t) for t in json.loads(Path("data/fresh/themes.json").read_text())} if Path("data/fresh/themes.json").exists() else set()
    seen = {key(t) for t in have} | reserved
    jobs = [(area, angle, w) for area in G.AREAS for angle in ANGLES for w in WRITERS]
    rng = random.Random(0)
    rng.shuffle(jobs)

    def one(job):
        area, angle, w = job
        try:
            text, _ = llm.chat(w, "You know development data catalogs well.", PROMPT.format(n=a.per, area=area, angle=angle),
                               temperature=0.95, max_tokens=1500, base_urls=WRITERS[w])
            return [t.strip() for t in llm.parse_json(text) if isinstance(t, str) and 3 <= len(t.strip()) <= 70]
        except Exception as e:
            print("skip", area[:20], angle[:20], w, type(e).__name__, flush=True)
            return []

    added = 0
    with ThreadPoolExecutor(24) as ex:
        for ts in ex.map(one, jobs):
            for t in ts:
                k = key(t)
                if k and k not in seen:
                    seen.add(k)
                    have.append(t)
                    added += 1
    out.write_text(json.dumps(have, ensure_ascii=False, indent=0))
    print(f"themes: +{added}, {len(have)} total (backup of the original at {backup})")


if __name__ == "__main__":
    main()
