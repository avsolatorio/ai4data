"""Coverage analytics over the realistic-search data, and sampling weights that steer the next chunks.

Reads data/real/c*/examples.jsonl (narrated pairs) and reports coverage by language, style, surface, writer, source,
relation, verdict, geography kind, country, grouping, time kind, population group and data type; catalog-item reuse;
search diversity (distinct word bigrams, exact duplicates, most common opening words, length by style). Writes
data/real/coverage.json with multiplicative weights for the generator:
  country, theme     inverse square root of the count so far (unseen ones get the largest weight)
  language, source   ratio of the target share to the observed share
  relation           steers anchored relations toward the target verdict mix

  python -m relevance_explainer.synth.coverage
"""
from __future__ import annotations

import argparse
import json
import math
import re
from collections import Counter
from pathlib import Path

from .realistic import LANG_W, RELATION
from . import generate as G

TARGET_VERDICT = {"relevant": 0.2, "partly_relevant": 0.32, "irrelevant": 0.48}
# Relations that mostly produce each verdict (from the pilot), used to steer the verdict mix.
REL_VERDICT = {"full": "relevant", "wrong_place": "irrelevant", "wrong_time": "partly_relevant", "wrong_group": "partly_relevant",
               "related": "partly_relevant", "other_topic": "irrelevant", "unrelated": "irrelevant"}


def entropy_ratio(c: Counter, k: int) -> float:
    n = sum(c.values())
    if not n or k <= 1:
        return 0.0
    h = -sum(v / n * math.log(v / n) for v in c.values() if v)
    return h / math.log(k)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="data/real")
    ap.add_argument("--geo", default="data/geo/geo.json")
    ap.add_argument("--themes", default="data/themes.json")
    a = ap.parse_args()
    root = Path(a.root)
    rows = []
    for p in sorted(root.glob("c*/examples.jsonl")):
        rows += [json.loads(l) for l in p.read_text().splitlines()]
    if not rows:
        print("no data yet")
        return
    geo = json.loads(Path(a.geo).read_text())
    themes = json.loads(Path(a.themes).read_text()) if Path(a.themes).exists() else []
    n = len(rows)
    rep = {"examples": n}
    for key in ("language", "style", "surface", "writer", "mode", "relation", "verdict"):
        rep[key] = dict(Counter(r.get(key) or "-" for r in rows).most_common())
    rep["source"] = dict(Counter(r["record"]["kind"] for r in rows).most_common())
    geo_kind = Counter((r.get("geo") or {}).get("kind", "none") for r in rows)
    rep["geo_kind"] = dict(geo_kind.most_common())
    countries = Counter()
    groups = Counter()
    sub = 0
    for r in rows:
        for pl in r.get("parsed", {}).get("places", []):
            if pl["kind"] == "country":
                countries[pl["iso3"]] += 1
            elif pl["kind"] == "group":
                groups[pl["code"]] += 1
            else:
                sub += 1
                countries[pl["iso3"]] += 0  # subnational places count toward the country's presence below
    allc = list(geo["countries"])
    rep["countries_mentioned"] = f"{sum(1 for c in allc if countries[c] > 0)} of {len(allc)}"
    rep["country_entropy_ratio"] = round(entropy_ratio(countries, len(allc)), 3)
    rep["countries_least"] = [geo["countries"][c]["name"] for c in sorted(allc, key=lambda c: countries[c])[:15]]
    rep["countries_most"] = [(geo["countries"][c]["name"], v) for c, v in countries.most_common(10)]
    rep["groups_mentioned"] = f"{len(groups)} of {len(geo['groups'])}"
    rep["subnational_mentions"] = sub
    rep["time_kind"] = dict(Counter((r.get("time") or {}).get("kind", "none") for r in rows).most_common())
    rep["group"] = dict(Counter(r.get("group") or "none" for r in rows).most_common())
    rep["datatype"] = dict(Counter(r.get("datatype") or "none" for r in rows).most_common())
    items = Counter(r["record"]["id"] for r in rows)
    rep["items_unique"] = len(items)
    rep["items_reused_5plus"] = sum(1 for v in items.values() if v >= 5)
    qs = [r["query"] for r in rows]
    uq = set(q.lower() for q in qs)
    rep["searches_unique"] = f"{len(uq)} of {len(qs)}"
    big = Counter()
    for q in uq:
        w = re.findall(r"\w+", q.lower())
        big.update(zip(w, w[1:]))
    rep["distinct_bigram_ratio"] = round(len(big) / max(1, sum(big.values())), 3)
    rep["opening_words"] = Counter(" ".join(re.findall(r"\w+", q.lower())[:2]) for q in uq).most_common(12)
    rep["words_by_style"] = {s: round(sum(len(r["query"].split()) for r in rows if r.get("style") == s) / max(1, sum(1 for r in rows if r.get("style") == s)), 1)
                             for s in ("keywords2", "keywords", "phrase", "question", "long")}
    rep["verdict_by_relation"] = {rel: dict(Counter(r["verdict"] for r in rows if r.get("relation") == rel)) for rel in RELATION}
    theme_c = Counter(r.get("topic") for r in rows)
    rep["themes_used"] = f"{sum(1 for t in themes if theme_c[t] > 0)} of {len(themes)}"

    # weights for the next chunks
    w = {}
    w["country"] = {c: round(1 / math.sqrt(1 + countries[c]), 4) for c in allc}
    m = max(w["country"].values())
    w["country"] = {c: round(v / m, 4) for c, v in w["country"].items()}
    w["theme"] = {t: round(1 / math.sqrt(1 + theme_c[t]), 4) for t in themes}
    lang = Counter(r["language"] for r in rows)
    tot_w = sum(LANG_W.values())
    w["language"] = {l: round(min(3.0, (LANG_W[l] / tot_w) / max(1e-6, lang[l] / n)), 3) for l in LANG_W}
    src = Counter(r["record"]["kind"] for r in rows)
    tot_s = sum(G.SOURCE_WEIGHTS.values())
    kind_of = {"indicator": "indicator", "study": "study", "document": "document", "dataset": "dataset"}
    w["source"] = {s: round(min(3.0, (G.SOURCE_WEIGHTS[s] / tot_s) / max(1e-6, src.get(kind_of.get(s, s), 0) / n)), 3) for s in G.SOURCE_WEIGHTS}
    ver = Counter(r["verdict"] for r in rows)
    ver_ratio = {v: TARGET_VERDICT[v] / max(1e-6, ver[v] / n) for v in TARGET_VERDICT}
    w["relation"] = {rel: round(min(3.0, max(0.3, ver_ratio[REL_VERDICT[rel]])), 3) for rel in RELATION}
    (root / "coverage.json").write_text(json.dumps({"report": rep, "weights": w}, ensure_ascii=False, indent=1))
    for k, v in rep.items():
        print(f"{k}: {v}")
    print("weights: language", w["language"], "| source", w["source"], "| relation", w["relation"])


if __name__ == "__main__":
    main()
