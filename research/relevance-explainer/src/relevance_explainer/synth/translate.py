"""Translation augmentation: narrated realistic pairs re-expressed in another language.

For each selected pair, a model translates the search and the narrative together into a target language (searches
keep their style: keywords stay keywords, typos are not corrected away, place names follow local usage). The verdict
is kept; the input is re-rendered with query understanding on the translated search; the similarity is recomputed.
Both texts must pass the language check. Output in the realistic example schema under data/real/tN/.

  python -m relevance_explainer.synth.translate --src data/real/c1 data/real/c2 --out data/real/t1 --n 4000
"""
from __future__ import annotations

import argparse
import json
import random
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from . import llm
from .realistic import LANG_W, language_ok, render_v2, render_v2min
from .understand import Understander

ROUTE = {"Swahili": ("gemma-4-31b", ["http://localhost:8002/v1"]), "Hindi": ("gemma-4-31b", ["http://localhost:8002/v1"]),
         "Indonesian": ("gemma-4-31b", ["http://localhost:8002/v1"]), "Turkish": ("gemma-4-31b", ["http://localhost:8002/v1"])}
DEFAULT = ("qwen3.8-27b", ["http://localhost:8000/v1", "http://localhost:8001/v1"])
PROMPT = """Translate a data-catalog search and the explanation shown for one search result from {src} into {tgt}.

Search: {query}
Explanation: {narrative}

Rules: keep the search's form (if it is a few keywords, write a few keywords; keep its level of formality; a casual search stays casual). Write place names as {tgt} speakers normally write them. Keep item titles, database names, codes and numbers exactly as in the explanation. Translate the explanation faithfully without adding or removing facts.

Return only a JSON object: {{"search": "...", "explanation": "..."}}"""


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", nargs="+", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--n", type=int, default=4000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--workers", type=int, default=64)
    a = ap.parse_args()
    rng = random.Random(a.seed)
    rows = []
    for d in a.src:
        rows += [json.loads(l) for l in (Path(d) / "examples.jsonl").read_text().splitlines()]
    have = Counter(r["language"] for r in rows)
    langs = list(LANG_W)
    # target languages weighted toward those with fewer examples relative to their target share
    tw = {l: LANG_W[l] / max(1, have[l]) for l in langs}
    pick = rng.sample(rows, min(a.n, len(rows)))
    jobs = []
    for r in pick:
        cands = [l for l in langs if l != r["language"]]
        jobs.append((r, rng.choices(cands, weights=[tw[l] for l in cands])[0]))

    def one(job):
        r, tgt = job
        model, urls = ROUTE.get(tgt, DEFAULT)
        try:
            text, _ = llm.chat(model, "You are a professional translator.", PROMPT.format(src=r["language"], tgt=tgt, query=r["query"], narrative=r["narrative"]),
                               temperature=0.3, max_tokens=700, base_urls=urls)
            j = llm.parse_json(text)
            q, n = str(j["search"]).strip(), str(j["explanation"]).strip()
        except Exception:
            return None
        if not q or not n or "\n" in n or not language_ok(q, tgt) or not language_ok(n, tgt):
            return None
        return {**r, "query": q, "query_written": q, "narrative": n, "language": tgt, "translated_from": r["language"], "translator": model}

    with ThreadPoolExecutor(a.workers) as ex:
        out = [x for x in ex.map(one, jobs) if x]
    u = Understander()
    from .signals import item_text, similarities
    sims = similarities([r["query"] for r in out], [item_text(r["record"]) for r in out])
    for r, s in zip(out, sims):
        r["parsed"] = u.parse(r["query"], r.get("facet_values"))
        r["similarity"] = float(s)
        r["input_v2"] = render_v2(r, u, r["parsed"], float(s))
        r["input_v2min"] = render_v2min(r, float(s))
        r["mode"] = "translated"
    d = Path(a.out)
    d.mkdir(parents=True, exist_ok=True)
    (d / "examples.jsonl").write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in out))
    print(f"translated {len(out)} of {len(jobs)}; targets {dict(Counter(r['language'] for r in out))}")


if __name__ == "__main__":
    main()
