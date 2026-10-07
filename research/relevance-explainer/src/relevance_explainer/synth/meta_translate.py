"""Catalog metadata in other languages: realistic pairs whose catalog record is translated into the language a national
statistical office would publish in, then narrated again by the teacher.

For each target metadata language L, about 70% of the selected pairs have a search already written in L and 30% have an
English search (international users of a national catalog). The record's descriptive text (title, description,
definition, topics, tags, geographic scope and similar) is translated; organisation names, codes, numbers and the
structured country list stay as they are (the metadata-schemas adapter maps country names in any language to canonical
names, so the Coverage line is the same at inference). The input is rendered with render version 2 and narrated with
NARRATE_V2, so verdict and narrative are new teacher outputs. Pool records published natively in another language
(Statistics Canada French, ILOSTAT French and Spanish) enter through the record pools instead.

  python -m relevance_explainer.synth.meta_translate --src data/real/c81 data/real/c82 --out data/real_meta/m1 --n 3000
"""
from __future__ import annotations

import argparse
import json
import random
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from . import generate as G
from . import llm
from .realistic import NARRATE_V2, RENDER_VERSION, render_v2, render_v2min
from .understand import Understander

META_LANG_W = {"French": 25, "Spanish": 25, "Portuguese": 14, "Arabic": 14, "Russian": 10, "Indonesian": 6, "Chinese": 6}
TEXT_FIELDS = ("name", "abstract", "definition", "concept", "limitation", "notes", "passage", "geography", "universe",
               "unit_of_analysis", "data_kind", "collection_mode", "doc_type", "unit", "columns", "rows", "purpose", "datasets")
LIST_FIELDS = ("topics", "tags")
PROMPT = """Translate the descriptive text of a data-catalog record from English into {tgt}, as the national statistical office of a {tgt}-speaking country would publish it.

Record fields (JSON):
{fields}

Rules: translate every value into natural, formal {tgt} catalog language. Keep organisation names, survey acronyms, codes, numbers, years and units of measure as they are, unless an official {tgt} name is standard. Keep the same keys. Keep lists as lists of the same length. Do not add or remove information.

Return only the JSON object with the translated values."""
DEFAULT = ("qwen3.8-27b", ["http://localhost:8000/v1", "http://localhost:8001/v1"])
ROUTE = {"Indonesian": ("gemma-4-31b", ["http://localhost:8002/v1"])}


def translate_record(rec: dict, tgt: str) -> dict | None:
    fields = {k: rec[k] for k in TEXT_FIELDS if isinstance(rec.get(k), str) and rec[k].strip()}
    fields.update({k: rec[k] for k in LIST_FIELDS if rec.get(k)})
    if not fields:
        return None
    model, urls = ROUTE.get(tgt, DEFAULT)
    text, _ = llm.chat(model, "You translate statistical metadata precisely.", PROMPT.format(tgt=tgt, fields=json.dumps(fields, ensure_ascii=False, indent=0)),
                       temperature=0.2, max_tokens=2500, base_urls=urls)
    j = llm.parse_json(text)
    if set(j) != set(fields):
        return None
    for k in LIST_FIELDS:
        if k in fields and (not isinstance(j[k], list) or len(j[k]) != len(fields[k])):
            return None
    out = dict(rec)
    for k, v in j.items():
        out[k] = [str(x) for x in v] if k in LIST_FIELDS else str(v)
    out["meta_language"] = tgt
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", nargs="+", required=True, help="realistic chunk directories with examples.jsonl")
    ap.add_argument("--out", required=True)
    ap.add_argument("--n", type=int, default=3000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--workers", type=int, default=48)
    a = ap.parse_args()
    rng = random.Random(a.seed)
    by_lang = defaultdict(list)
    for d in a.src:
        for l in (Path(d) / "examples.jsonl").read_text().splitlines():
            r = json.loads(l)
            if r["record"]["kind"] in ("indicator", "study", "document", "dataset", "table", "geospatial", "script"):
                by_lang[r["language"]].append(r)
    for v in by_lang.values():
        rng.shuffle(v)
    jobs = []
    tot = sum(META_LANG_W.values())
    for L, w in META_LANG_W.items():
        n = round(a.n * w / tot)
        same, eng = by_lang.get(L, []), by_lang.get("English", [])
        k_same = min(len(same), round(n * 0.7))
        jobs += [(r, L) for r in same[:k_same]]
        jobs += [(r, L) for r in rng.sample(eng, min(len(eng), n - k_same))]
    u = Understander()

    def one(job):
        r, L = job
        rec = translate_record(r["record"], L)
        if rec is None:
            return []
        p = {k: v for k, v in r.items() if k not in ("verdict", "narrative", "missed", "input_v2", "input_v2min", "input_v1", "input_v1x")}
        p.update(record=rec, render_version=RENDER_VERSION, origin="metatrans", meta_language=L)
        parsed = u.parse(p["query"], p.get("facet_values"), level=p["render_version"])
        p["parsed"] = parsed
        p["input_v2"] = render_v2(p, u, parsed, p["similarity"])
        p["input_v2min"] = render_v2min(p, p["similarity"])
        text, _ = llm.chat(G.NARRATE_MODEL, NARRATE_V2, p["input_v2"], temperature=0.4, max_tokens=450)
        j = llm.parse_json(text)
        if j.get("verdict") not in ("relevant", "partly_relevant", "irrelevant"):
            return []
        return [{**p, "verdict": j["verdict"], "narrative": str(j["narrative"]).strip(), "missed": j.get("missed") or [],
                 "verdict_english_metadata": r["verdict"]}]

    with ThreadPoolExecutor(a.workers) as ex:
        rows = [x for xs in ex.map(lambda jb: G._safe(one, jb), jobs) for x in xs]
    Path(a.out).mkdir(parents=True, exist_ok=True)
    (Path(a.out) / "examples.jsonl").write_text("".join(json.dumps(x, ensure_ascii=False) + "\n" for x in rows))
    same = sum(x["verdict"] == x["verdict_english_metadata"] for x in rows) / max(1, len(rows))
    print(f"metadata-translated: {len(rows)} of {len(jobs)}; verdict unchanged from the English-metadata label: {same:.3f}")


if __name__ == "__main__":
    main()
