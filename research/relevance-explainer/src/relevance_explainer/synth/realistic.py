"""Realistic search data: code samples a diverse spec, the teacher writes the search, the pipeline renders inputs.

Stages (resumable; files in --out):
  queries   theme searches: sample a spec per search, the teacher writes the search text and English keywords
  anchored  record-anchored searches: pick a catalog record and a relation (fully answers, wrong place, wrong time,
            wrong group, related concept, same place but another topic, unrelated); code picks constraint values
            consistent with the relation and the record's coverage; the teacher writes the search
  retrieve  catalog search for theme searches (ranks 1, one of 2-8, one of 9-30); anchored searches keep their record
  render    query understanding and checks; inputs in the new format (v2) and in the first rounds' format (v1, with
            only interface facets as filters) and v1 with extracted constraints as filters (v1x); retrieval similarity
  narrate   the teacher writes verdict and narrative from the v2 input, and lists constraints the extractor missed

See docs/design_realistic.md.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import random
import re
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from . import generate as G
from . import llm
from .geo import LANG_CODES
from .sources import SOURCES, filter_check
from .understand import GROUP_EVIDENCE, Understander, _record_iso3, _record_text, check, time_label

LANG_W = {"English": 14, "French": 9, "Spanish": 9, "Portuguese": 8, "Arabic": 9, "Chinese": 9, "Russian": 9,
          "Indonesian": 8, "Hindi": 9, "Turkish": 8, "Swahili": 8}
STYLE = {"keywords2": (20, "one or two keywords only, no grammar (for example: 'malaria nampula')"),
         "keywords": (35, "a keyword string of 3 to 7 words without full grammar, as typed into a search box"),
         "phrase": (25, "a short natural phrase (for example: 'unemployment among young women in Tunisia')"),
         "question": (15, "a full question or request in one sentence"),
         "long": (5, "a long, specific request of 2 sentences or 25 to 40 words describing exactly what data is needed and why")}
SURFACE = {"clean": 70, "typos": 15, "codeswitch": 15}
GEO_KIND = {"none": 25, "country": 35, "multi": 8, "wb_group": 8, "other_group": 6, "subnational": 12, "global": 6}
TIME_KIND = {"none": 40, "year": 20, "range": 12, "decade": 5, "relative": 18, "round": 5}
GROUPS = list(GROUP_EVIDENCE)
DATATYPE = {"none": 55, "indicator": 12, "microdata": 13, "document": 10, "dataset": 10}
FACETS = {"none": 75, "duplicate": 10, "extra": 15}
RELATION = {"full": 30, "wrong_place": 12, "wrong_time": 10, "wrong_group": 6, "related": 14, "other_topic": 12, "unrelated": 16}
RELATION_TEXT = {
    "full": "a search that this record answers well: its topic, and any place, time or group mentioned, are covered by the record",
    "wrong_place": "a search on the record's topic, but for the place given below, which the record does not cover",
    "wrong_time": "a search on the record's topic, but for the period given below, which the record does not cover",
    "wrong_group": "a search on the record's topic, but for the population group given below, which the record does not mention",
    "related": "a search for a related but different concept than what the record measures or contains (for example a neighbouring indicator, a cause or consequence, a narrower or broader measure)",
    "other_topic": "a search on a clearly different topic, for the place given below that the record covers",
    "unrelated": "a search on a topic unrelated to the record",
}


# Search writers: several models give more varied searches. QUERY_WRITERS='{"model": {"urls": [...], "share": 0.6}, ...}'.
# Narratives and labels stay with one teacher (NARRATE_MODEL) for consistent verdicts.
WRITERS = json.loads(os.environ.get("QUERY_WRITERS") or (Path("data/real/writers.json").read_text() if Path("data/real/writers.json").exists() else "{}"))


def writer(key: str) -> tuple[str, list[str] | None]:
    """Model and server list for one search, drawn by its id so reruns pick the same writer."""
    if not WRITERS:
        return G.QUERY_MODEL, None
    names = list(WRITERS)
    m = random.Random(f"writer-{key}").choices(names, weights=[WRITERS[n].get("share", 1.0) for n in names])[0]
    return m, WRITERS[m].get("urls")


SCRIPT = {"Arabic": r"[؀-ۿ]", "Chinese": r"[一-鿿]", "Hindi": r"[ऀ-ॿ]", "Russian": r"[Ѐ-ӿ]"}
NEAR = {"Indonesian": {"ms"}, "Portuguese": {"gl"}, "Spanish": {"gl", "ca"}}
LID_PATH = Path("data/geo/lid.176.ftz")
_LID = None


def language_ok(q: str, lang: str) -> bool:
    """The search is in the requested language: target script for Arabic, Chinese, Hindi and Russian; for the others
    fastText language ID must not confidently name another language (searches under 4 words are not checked)."""
    global _LID
    if lang in SCRIPT:
        return re.search(SCRIPT[lang], q) is not None
    if len(re.findall(r"\w+", q)) < 4 or not LID_PATH.exists():
        return True
    if _LID is None:
        import fasttext
        _LID = fasttext.load_model(str(LID_PATH))
    p, label = _LID.f.predict(q.replace("\n", " "), 1, 0.0, "strict")[0]
    code = label.replace("__label__", "")
    return code == LANG_CODES[lang] or code in NEAR.get(lang, set()) or p < 0.6 or (code == "en" and p < 0.8)


def recase(q: str, rng: random.Random) -> str:
    """A few searches typed in capitals or Title Case, as in real logs."""
    x = rng.random()
    if x < 0.03:
        return q.upper()
    if x < 0.08:
        return " ".join(w[:1].upper() + w[1:] for w in q.split())
    return q


def budget(spec: dict, rng: random.Random) -> None:
    """Short searches cannot carry many constraints: two keywords keep one, a keyword string at most three."""
    cap = {"keywords2": 1, "keywords": 3}.get(spec["style"], 4)
    have = [k for k in ("geo", "time", "group", "datatype") if spec.get(k) and spec.get(k) != "none"]
    rng.shuffle(have)
    for k in sorted(have, key=lambda k: k not in ("geo", "time"))[cap:]:  # place and time are kept first
        spec[k] = "none" if k == "datatype" else None


# Coverage feedback (synth.coverage): multiplicative weights that steer sampling toward under-covered cells.
COV_PATH = Path("data/real/coverage.json")
COV = json.loads(COV_PATH.read_text()).get("weights", {}) if COV_PATH.exists() else {}


def cov(kind: str, key: str) -> float:
    return float(COV.get(kind, {}).get(key, 1.0))


def pick_theme(rng: random.Random, themes: list[str]) -> str:
    return rng.choices(themes, weights=[cov("theme", t) for t in themes])[0] if COV.get("theme") else rng.choice(themes)


def pick_source(rng: random.Random) -> str:
    keys = list(G.SOURCE_WEIGHTS)
    return rng.choices(keys, weights=[G.SOURCE_WEIGHTS[k] * cov("source", k) for k in keys])[0]


def wpick(rng: random.Random, d: dict):
    keys = list(d)
    return rng.choices(keys, weights=[v[0] if isinstance(v, tuple) else v for v in d.values()])[0]


class Specs:
    def __init__(self, geo: dict, seed: int):
        self.geo, self.rng = geo, random.Random(seed)
        self.countries = sorted(geo["countries"])
        self.places_by = {}
        for p in geo["places"]:
            self.places_by.setdefault(p["iso3"], []).append(p)
        g = geo["groups"]
        self.wb_groups = [k for k, v in g.items() if v["kind"] in ("wb_region", "income") and k not in ("EUU", "EMU", "OED", "LDC", "SST", "PSS", "CSS", "HPC", "ARB")]
        self.other_groups = [k for k, v in g.items() if v["kind"] == "curated"] + [k for k in ("EUU", "OED", "LDC", "SST", "PSS", "CSS", "HPC", "ARB") if k in g]

        # Country weight: half uniform, half by size (log of the number of large cities), so small territories still
        # appear but less often than in a uniform draw.
        ncity = {c: sum(1 for p in self.places_by.get(c, []) if p["kind"] == "city") for c in self.countries}
        self.cweight = {c: (0.5 + 0.5 * (1 + math.log1p(ncity[c]))) * cov('country', c) for c in self.countries}

    def cname(self, iso3: str) -> str:
        return self.geo["countries"][iso3]["name"]

    def _country(self, pool: list[str]) -> str:
        return self.rng.choices(pool, weights=[self.cweight.get(c, 1.0) for c in pool])[0]

    def geography(self, kind: str, allowed: list[str] | None = None, avoid: set | None = None) -> dict | None:
        r, pool = self.rng, [c for c in (allowed or self.countries) if not avoid or c not in avoid]
        if kind == "none" or not pool:
            return None
        if kind == "country":
            c = self._country(pool)
            return {"kind": "country", "iso3": [c], "text": self.cname(c)}
        if kind == "multi":
            cs = list(dict.fromkeys(self._country(pool) for _ in range(r.choice([2, 2, 3]))))
            if len(cs) < 2:
                cs = r.sample(pool, min(len(pool), 2))
            return {"kind": "multi", "iso3": cs, "text": " and ".join(self.cname(c) for c in cs)}
        if kind in ("wb_group", "other_group"):
            code = r.choice(self.wb_groups if kind == "wb_group" else self.other_groups)
            return {"kind": "group", "code": code, "text": self.geo["groups"][code]["name"]}
        if kind == "subnational":
            for _ in range(20):
                c = self._country(pool)
                if self.places_by.get(c):
                    ps = self.places_by[c]  # provinces evenly, cities by the square root of population
                    p = r.choices(ps, weights=[1.0 if q["kind"] == "admin1" else max(1.0, (q["population"] / 150000) ** 0.5) for q in ps])[0]
                    what = "city" if p["kind"] == "city" else "province, state or region"
                    return {"kind": "subnational", "iso3": [c], "place": p["name"], "text": f"{p['name']} ({what} in {self.cname(c)})"}
            return None
        return {"kind": "global", "text": r.choice(["all countries / global", "developing countries", "worldwide, by country"])}

    def time(self, kind: str, inside: tuple | None = None, outside: tuple | None = None) -> dict | None:
        r = self.rng
        lo, hi = inside or (1990, 2025)
        if outside:
            a, b = outside
            choices = [y for y in range(1985, 2026) if y < a or y > b]
            if not choices:
                return None
            y = r.choice(choices)
            return {"kind": "year", "text": str(y), "start": y, "end": y}
        if kind == "none":
            return None
        if kind == "year":
            y = r.randint(lo, hi) if r.random() < 0.4 else max(lo, min(hi, r.randint(2010, 2025)))
            return {"kind": "year", "text": str(y), "start": y, "end": y}
        if kind == "range":
            a = r.randint(lo, max(lo, hi - 2))
            b = r.randint(a + 1, min(hi, a + 15)) if hi > a else a
            return {"kind": "range", "text": f"{a} to {b}", "start": a, "end": b}
        if kind == "decade":
            d = r.choice([d for d in (1990, 2000, 2010, 2020) if lo <= d + 9 and d <= hi] or [2010])
            return {"kind": "decade", "text": f"the {d}s", "start": d, "end": d + 9}
        if kind == "relative":
            k = r.choice(["latest", "recent", "since", "last", "pre_covid", "post_covid"])
            if k == "since":
                y = r.randint(max(lo, 2000), max(lo, min(hi, 2022)))
                return {"kind": "since", "text": f"since {y}", "start": y, "end": 2026}
            if k == "last":
                n = r.choice([3, 5, 10])
                return {"kind": "last", "text": f"the last {n} years", "start": 2026 - n, "end": 2026}
            return {"kind": k, "text": {"latest": "the latest / most recent data", "recent": "recent years",
                                         "pre_covid": "before COVID-19", "post_covid": "since the COVID-19 pandemic"}[k]}
        y = r.randint(max(lo, 1995), max(max(lo, 1995), min(hi, 2023)))
        return {"kind": "round", "text": f"a named survey round such as '{r.choice(['DHS', 'MICS', 'LSMS', 'census', 'labour force survey', 'household budget survey'])} {y}'", "start": y, "end": y}

    def base(self) -> dict:
        r = self.rng
        lang = wpick(r, {k: v * cov('language', k) for k, v in LANG_W.items()})
        surface = wpick(r, SURFACE)
        if surface == "codeswitch" and lang == "English":
            surface = "clean"
        return {"language": lang, "style": wpick(r, STYLE), "surface": surface,
                "group": r.choice(GROUPS) if r.random() < 0.4 else None, "datatype": wpick(r, DATATYPE), "facets": wpick(r, FACETS)}


SURFACE_TEXT = {"clean": "spelled correctly", "typos": "with one or two realistic typos or missing accents, possibly all lowercase",
                "codeswitch": "mixing in one or two English technical terms, as people often do (for example 'GDP', 'microdata', 'dataset')"}

QUERY_PROMPT = """Write ONE search that a person would type into the search box of a development data catalog (statistical indicators, survey microdata, research papers, humanitarian datasets).

Topic: {topic}
Language: {language}
Form: {style}; {surface}.
The search must express these, in the person's own words (local names, abbreviations or informal forms are fine):
{must}
Do not add any other place, year or population group. Do not mention filters.

Return only a JSON object: {{"query": "...", "search_en": "3 to 6 English keywords for a catalog keyword search", "expressed": ["place", "time", "group", "type"]}} where "expressed" lists which of the required elements the search contains."""

ANCHOR_PROMPT = """A catalog record is shown below. Write ONE search that a person would type into the search box of a development data catalog, so that the search is {relation}.

Record ({label}):
{item}

Language: {language}
Form: {style}; {surface}.
{must}
Do not mention filters. Write as a user who has not seen the record.

Return only a JSON object: {{"query": "...", "expressed": ["place", "time", "group", "type"]}}"""

DT_TEXT = {"indicator": "an indicator or time series", "microdata": "survey microdata or census data", "document": "a paper or report", "dataset": "a dataset or data file"}


def must_lines(spec: dict) -> str:
    out = []
    if spec.get("geo"):
        out.append(f"- place: {spec['geo']['text']}")
    if spec.get("time"):
        out.append(f"- time: {spec['time']['text']}")
    if spec.get("group"):
        out.append(f"- population group: {spec['group']}")
    if spec.get("datatype") and spec["datatype"] != "none":
        out.append(f"- kind of data wanted: {DT_TEXT[spec['datatype']]}")
    return "\n".join(out) or "- nothing beyond the topic (no place, time or group)"


def facets_for(spec: dict, rng: random.Random, sp: Specs) -> dict:
    """Interface facets: none, a copy of a constraint in the text, or an extra constraint not in the text."""
    f = {}
    if spec["facets"] == "duplicate":
        if spec.get("geo") and spec["geo"].get("iso3"):
            f["countries"] = [sp.cname(c) for c in spec["geo"]["iso3"]]
        elif spec.get("time") and spec["time"].get("start"):
            f["years"] = str(spec["time"]["start"]) if spec["time"]["start"] == spec["time"].get("end") else f"{spec['time']['start']}-{spec['time']['end']}"
    elif spec["facets"] == "extra":
        if not spec.get("geo"):
            f["countries"] = [sp.cname(rng.choice(sp.countries))]
        elif not spec.get("time"):
            y = rng.randint(2005, 2024)
            f["years"] = str(y)
        else:
            f["frequency"] = rng.choice(["Annual", "Quarterly", "Monthly"])
    return f


# ---------------------------------------------------------------------- stages
def cmd_queries(a) -> None:
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    geo = json.loads(Path(a.geo).read_text())
    sp = Specs(geo, a.seed)
    themes = json.loads(Path(a.themes).read_text())
    specs = []
    for i in range(a.n):
        s = sp.base()
        s.update(id=f"q{a.seed}-{i}", mode="theme", topic=pick_theme(sp.rng, themes), source=pick_source(sp.rng))
        s["geo"] = sp.geography(wpick(sp.rng, GEO_KIND))
        s["time"] = sp.time(wpick(sp.rng, TIME_KIND))
        budget(s, sp.rng)
        s["facet_values"] = facets_for(s, sp.rng, sp)
        specs.append(s)

    def one(s: dict) -> dict | None:
        rng = random.Random(s["id"])
        for _ in range(3):  # retry searches written in the wrong language
            wm, wurls = writer(s["id"])
            text, _ = llm.chat(wm, G.QUERY_SYSTEM, QUERY_PROMPT.format(topic=s["topic"], language=s["language"], style=STYLE[s["style"]][1],
                                                                     surface=SURFACE_TEXT[s["surface"]], must=must_lines(s)), temperature=0.9, max_tokens=400, base_urls=wurls)
            j = llm.parse_json(text)
            q = str(j["query"]).strip()
            if q and language_ok(q, s["language"]):
                return {**s, "writer": wm, "query": recase(q, rng), "query_written": q, "search_en": str(j.get("search_en") or s["topic"]), "expressed": j.get("expressed", [])}
        return None

    with ThreadPoolExecutor(a.workers) as ex:
        rows = [r for r in ex.map(lambda s: G._safe(lambda x: [one(x)], s), specs) for r in r if r]
    (out / "queries.jsonl").write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows))
    print(f"theme searches: {len(rows)} of {len(specs)}")


def _record_years(rec: dict) -> tuple | None:
    return (rec["year_start"], rec["year_end"]) if rec.get("year_start") and rec.get("year_end") else None


def cmd_anchored(a) -> None:
    out = Path(a.out)
    geo = json.loads(Path(a.geo).read_text())
    sp = Specs(geo, a.seed + 1)
    themes = json.loads(Path(a.themes).read_text())
    rng = random.Random(a.seed + 2)
    draws = [(pick_theme(rng, themes), pick_source(rng)) for _ in range(a.n)]

    def one(i_draw):
        i, (theme, kind) = i_draw
        r = random.Random(f"{a.seed}-{i}")
        src = SOURCES[kind]
        hits = src.search(theme, 20)
        r.shuffle(hits)
        rec = None
        for h in hits[:6]:
            rec = src.hydrate(h, theme)
            if rec and G._informative(rec, r, 0.1):
                break
            rec = None
        if not rec:
            return None
        s = sp.base()
        rel = wpick(r, {k: v * cov('relation', k) for k, v in RELATION.items()})
        covered = sorted(_record_iso3(rec))
        yrs = _record_years(rec)
        s.update(id=f"a{a.seed}-{i}", mode="anchored", relation=rel, topic=theme, source=kind)
        if rel == "full":
            s["geo"] = sp.geography(r.choice(["none", "country", "country", "subnational"]), allowed=covered[:40] if 0 < len(covered) <= 60 else None) if covered else None
            s["time"] = sp.time(r.choice(["none", "year", "range", "relative"]), inside=yrs) if yrs and rec["kind"] != "document" else None
            s["group"] = s["group"] if s["group"] and re.search(GROUP_EVIDENCE[s["group"]], _record_text(rec), re.I) else None
        elif rel == "wrong_place":
            s["geo"] = sp.geography(r.choice(["country", "country", "subnational", "multi"]), avoid=set(covered)) if covered else sp.geography("country")
            s["time"] = None
        elif rel == "wrong_time":
            s["time"] = sp.time("year", outside=yrs) if yrs and rec["kind"] != "document" else sp.time("year")
            s["geo"] = sp.geography("country", allowed=covered[:40]) if covered and r.random() < 0.5 else None
        elif rel == "wrong_group":
            missing = [g for g in GROUPS if not re.search(GROUP_EVIDENCE[g], _record_text(rec), re.I)]
            s["group"] = r.choice(missing) if missing else r.choice(GROUPS)
            s["geo"], s["time"] = None, None
        elif rel in ("related", "other_topic"):
            s["geo"] = sp.geography("country", allowed=covered[:40]) if covered and r.random() < 0.7 else None
            s["time"] = sp.time(r.choice(["none", "year", "relative"]), inside=yrs) if yrs and rec["kind"] != "document" and r.random() < 0.4 else None
        else:  # unrelated
            s["topic"] = r.choice(themes)
            s["geo"] = sp.geography(wpick(r, GEO_KIND))
            s["time"] = sp.time(wpick(r, TIME_KIND))
        s["facet_values"] = facets_for(s, r, sp)
        item = "\n".join(src.render(rec))
        must = must_lines(s)
        for _ in range(3):  # retry searches written in the wrong language
            wm, wurls = writer(s["id"])
            text, _ = llm.chat(wm, G.QUERY_SYSTEM, ANCHOR_PROMPT.format(relation=RELATION_TEXT[rel], label=src.label, item=item[:2500],
                                                                                 language=s["language"], style=STYLE[s["style"]][1],
                                                                                 surface=SURFACE_TEXT[s["surface"]],
                                                                                 must=("The search must express:\n" + must) if (s.get("geo") or s.get("time") or s.get("group")) else "Do not mention a place, year or population group unless the relation requires it."),
                               temperature=0.9, max_tokens=400, base_urls=wurls)
            j = llm.parse_json(text)
            q = str(j["query"]).strip()
            if q and language_ok(q, s["language"]):
                return {**s, "writer": wm, "query": recase(q, r), "query_written": q, "expressed": j.get("expressed", []), "record": rec, "rank": 0}
        return None

    with ThreadPoolExecutor(a.retrieve_workers) as ex:
        rows = [r for r in ex.map(lambda d: G._safe(lambda x: [one(x)], d), enumerate(draws)) for r in r if r]
    (out / "anchored.jsonl").write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows))
    print(f"anchored searches: {len(rows)} of {len(draws)}")


def cmd_retrieve(a) -> None:
    out = Path(a.out)
    rows = [json.loads(l) for l in (out / "queries.jsonl").read_text().splitlines()]
    rng = random.Random(a.seed)

    def one(q):
        src = SOURCES[q["source"]]
        hits = list(enumerate(src.search(q["search_en"], 30), 1))
        chosen = hits[:1] + rng.sample(hits[1:8], min(1, len(hits[1:8]))) + rng.sample(hits[8:], min(1, len(hits[8:])))
        res = []
        for rank, h in chosen:
            rec = src.hydrate(h, q["search_en"])
            if rec and G._informative(rec, rng):
                res.append({**q, "record": rec, "rank": rank})
        return res

    with ThreadPoolExecutor(a.retrieve_workers) as ex:
        pairs = [p for ps in ex.map(lambda q: G._safe(one, q), rows) for p in ps]
    if (out / "anchored.jsonl").exists():
        pairs += [json.loads(l) for l in (out / "anchored.jsonl").read_text().splitlines()]
    (out / "pairs.jsonl").write_text("".join(json.dumps(p, ensure_ascii=False) + "\n" for p in pairs))
    print(f"pairs: {len(pairs)}")


def _clean_item_lines(lines: list[str]) -> list[str]:
    """Drop fields the catalog does not have (empty or n/a)."""
    return [l for l in lines if not re.search(r":\s*(n/a|none|null|)\s*$", l.strip(), re.I)]


def render_v2(p: dict, u: Understander, parsed: dict, sim: float) -> str:
    src = SOURCES[p["record"]["kind"]]
    fl = "; ".join(f"{k}={'/'.join(v) if isinstance(v, list) else v}" for k, v in (p.get("facet_values") or {}).items()) or "none"
    lines = [f"Respond in: {p['language']}", f"Item type: {src.label}", f"Search: {p['query']}", f"Facets: {fl}",
             f"Understood: {u.describe(parsed)}"]
    chk = check(p["record"], parsed, u)
    lines.append("Checks: " + (" | ".join(chk) if chk else "none"))
    lines.append(f"Retrieval similarity: {sim:.2f}")
    lines += _clean_item_lines(src.render(p["record"]))
    return "\n".join(lines)


def item_first(v2: str) -> str:
    """Reorder a full-format input: the item block first (cacheable per item), then the search and facets, then the
    query understanding, checks and similarity that depend on both. Same lines, different order."""
    lines = v2.split("\n")
    head_keys = ("Respond in:", "Item type:", "Search:", "Facets:", "Understood:", "Checks:", "Retrieval similarity:")
    head = {l.split(":", 1)[0] + ":": l for l in lines if l.startswith(head_keys)}
    item = [l for l in lines if not l.startswith(head_keys)]
    order = [head.get(k) for k in ("Item type:",)] + item + [head.get(k) for k in ("Respond in:", "Search:", "Facets:", "Understood:", "Checks:", "Retrieval similarity:")]
    return "\n".join(l for l in order if l)


def render_v2min(p: dict, sim: float) -> str:
    """Minimal format: the search, facets, similarity and the record, with no query understanding or checks."""
    src = SOURCES[p["record"]["kind"]]
    fl = "; ".join(f"{k}={'/'.join(v) if isinstance(v, list) else v}" for k, v in (p.get("facet_values") or {}).items()) or "none"
    return "\n".join([f"Respond in: {p['language']}", f"Item type: {src.label}", f"Search: {p['query']}", f"Facets: {fl}",
                      f"Retrieval similarity: {sim:.2f}", *_clean_item_lines(src.render(p["record"]))])


def v1_filters(parsed: dict, u: Understander) -> dict:
    """Extracted constraints in the first rounds' filter vocabulary (countries, years, frequency)."""
    f = {}
    cs = [pl["name"] for pl in parsed["places"] if pl["kind"] == "country"]
    cs += [u.countries[pl["iso3"]]["name"] for pl in parsed["places"] if pl["kind"] in ("admin1", "city") and not pl.get("ambiguous")]
    if cs:
        f["countries"] = list(dict.fromkeys(cs))
    t = parsed.get("time")
    if t and t.get("start") and t.get("end") and t["type"] not in ("latest", "recent"):
        f["years"] = str(t["start"]) if t["start"] == t["end"] else f"{t['start']}-{min(t['end'], 2025)}"
    if parsed.get("frequency"):
        f["frequency"] = parsed["frequency"]["value"].title()
    return f


def render_v1(p: dict, filters: dict, sim: float) -> str:
    from .signals import add_similarity_line
    return add_similarity_line(G.render_input(p["query"], filters, p["record"], p["language"]), sim)


def cmd_render(a) -> None:
    from .signals import item_text, similarities
    out = Path(a.out)
    pairs = [json.loads(l) for l in (out / "pairs.jsonl").read_text().splitlines()]
    u = Understander()
    sims = similarities([p["query"] for p in pairs], [item_text(p["record"]) for p in pairs])
    rows = []
    for p, s in zip(pairs, sims):
        parsed = u.parse(p["query"], p.get("facet_values"))
        rows.append({**p, "parsed": parsed, "similarity": float(s),
                     "input_v2": render_v2(p, u, parsed, float(s)),
                     "input_v2min": render_v2min(p, float(s)),
                     "input_v1": render_v1(p, p.get("facet_values") or {}, float(s)),
                     "input_v1x": render_v1(p, {**(p.get("facet_values") or {}), **v1_filters(parsed, u)}, float(s))})
    (out / "rendered.jsonl").write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows))
    print(f"rendered: {len(rows)}")


# A typed-constraint rule (wrong place -> partly relevant) was tried and removed: in the first rounds a failed country
# filter goes with irrelevant in 96% of 61,717 training examples, so the teacher's convention is already consistent.
ORDER_RULE = (
    "- Order: first say what the item offers that fits the search (its topic, place, period or group, as far as they "
    "match), then the gaps or mismatches. For a relevant item, the fit is the whole narrative, with any minor caveat last. "
    "For an irrelevant item, say briefly what the item actually contains, then why it does not answer the search; do not "
    "invent a fit that is not there.")
NARRATE_V2 = G.NARRATE_SYSTEM.replace(
    "- For partly_relevant, say exactly which part of the query is satisfied and which is not.",
    ORDER_RULE + "\n- For partly_relevant, say exactly which part of the query is satisfied and which is not.",
).replace(
    "- Trust the \"Filter check\" line: do not contradict it.",
    "- The \"Understood\" line lists the place, time, group and data-type constraints found in the search and the facets; the \"Checks\" line gives each one's status against the record. Trust the checks: do not contradict them. Refer to constraints as the user expressed them (\"your search mentions 2015\"), not as filters, unless they came from a facet.\n- If the search contains a constraint that the Understood line missed, judge it from the metadata yourself.",
).replace('Return only a JSON object: {"verdict": "relevant" | "partly_relevant" | "irrelevant", "narrative": "..."}',
          'Return only a JSON object: {"verdict": "relevant" | "partly_relevant" | "irrelevant", "narrative": "...", "missed": ["place, time, population-group or data-type constraints in the search that the Understood line missed; not topic words; empty if none"]}')


def cmd_narrate(a) -> None:
    out = Path(a.out)
    rows = [json.loads(l) for l in (out / "rendered.jsonl").read_text().splitlines()]

    def one(r):
        text, _ = llm.chat(G.NARRATE_MODEL, NARRATE_V2, r["input_v2"], temperature=0.4, max_tokens=450)
        j = llm.parse_json(text)
        if j.get("verdict") not in ("relevant", "partly_relevant", "irrelevant"):
            return None
        return {**r, "verdict": j["verdict"], "narrative": str(j["narrative"]).strip(), "missed": j.get("missed") or []}

    with ThreadPoolExecutor(a.workers) as ex:
        done = [x for x in ex.map(lambda r: G._safe(lambda y: [one(y)], r), rows) for x in x if x]
    (out / "examples.jsonl").write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in done))
    print(f"narrated: {len(done)} of {len(rows)}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("stage", choices=["queries", "anchored", "retrieve", "render", "narrate"])
    ap.add_argument("--out", required=True)
    ap.add_argument("--n", type=int, default=400)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--themes", default="data/fresh/themes.json")
    ap.add_argument("--geo", default="data/geo/geo.json")
    ap.add_argument("--workers", type=int, default=96)
    ap.add_argument("--retrieve-workers", type=int, default=12)
    a = ap.parse_args()
    {"queries": cmd_queries, "anchored": cmd_anchored, "retrieve": cmd_retrieve, "render": cmd_render, "narrate": cmd_narrate}[a.stage](a)


if __name__ == "__main__":
    main()
