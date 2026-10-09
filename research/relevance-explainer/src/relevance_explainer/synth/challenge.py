"""Challenge suite: test searches for failure modes the training generator may not cover, written independently of it.

Each phenomenon has its own instruction. One writer (Gemma 4, a separate prompt from the training writers) writes one
search per (record, phenomenon) and states the verdict the case is designed for; the teacher narrates with NARRATE_V2
and Gemma 4 gives a second verdict. Records come from the realistic chunks (all four trained kinds) and the record pools.

  python -m relevance_explainer.synth.challenge write  --out data/challenge --per 150
  python -m relevance_explainer.synth.challenge label  --out data/challenge
  python -m relevance_explainer.synth.challenge build  --out data/challenge --eval data/eval_challenge
"""
from __future__ import annotations

import argparse
import json
import random
import re
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from . import generate as G
from . import llm
from .realistic import NARRATE_V2, NARRATE_V3, RENDER_VERSION, item_first, language_ok, render_v2, renderer, writer
from .understand import Understander

GEMMA = ("gemma-4-31b", ["http://localhost:8002/v1"])
TEACHER = ("qwen3.8-27b", ["http://localhost:8000/v1", "http://localhost:8001/v1"])
LANG_W = {"English": 50, "French": 8, "Spanish": 8, "Portuguese": 5, "Arabic": 6, "Chinese": 5, "Russian": 5, "Indonesian": 4,
          "Hindi": 3, "Turkish": 3, "Swahili": 3}
RENAMED = {"Eswatini": "Swaziland", "Myanmar": "Burma", "Congo, Dem. Rep.": "Zaire", "Czechia": "Czech Republic",
           "North Macedonia": "Macedonia", "Türkiye": "Turkey", "Turkiye": "Turkey", "Cabo Verde": "Cape Verde",
           "Timor-Leste": "East Timor", "Côte d'Ivoire": "Ivory Coast", "Cote d'Ivoire": "Ivory Coast", "Sri Lanka": "Ceylon",
           "Benin": "Dahomey", "Burkina Faso": "Upper Volta", "Zimbabwe": "Rhodesia", "Kyrgyz Republic": "Kirghizia",
           "Russian Federation": "Russia", "Viet Nam": "Vietnam", "Lao PDR": "Laos", "Iran, Islamic Rep.": "Persia"}

# name: (instruction, record filter, designed verdict)
PHENOMENA = {
    "synonym": ("The search asks for exactly what the record measures or contains, but in everyday words or a synonym, without the key terms of its title or definition (e.g. 'kids too short for their age' for stunting, 'how many people live in Kenya' for total population).", None, "relevant"),
    "acronym": ("The search uses an acronym, abbreviation or code for the concept, survey or indicator the record is about (e.g. 'DHS', 'LSMS', 'CPI', 'MPI', 'GDP pc', 'U5MR', an indicator code). Use one that people really use for this record's subject.", None, "relevant"),
    "standard_measure": ("The search names a broad concept for which this record is the standard or usual measure (e.g. 'inflation' for a consumer price index, 'literacy' for a literacy rate, 'extreme poverty' for a poverty headcount at the international line).", lambda r: r["kind"] in ("indicator", "table"), "relevant"),
    "proxy": ("The search asks for a concept that this record measures only indirectly or approximately (e.g. 'household income' for GDP per capita, 'unemployment' for labor force participation). The record is a proxy, not the thing asked.", lambda r: r["kind"] in ("indicator", "table", "study"), "partly_relevant"),
    "threshold": ("The search asks for the record's concept but with a different threshold, unit, age band or definition than the record uses (e.g. '$3 a day poverty' for a $1.90 line, 'under 3' for an under-5 measure, a count when the record gives a percentage).", lambda r: r["kind"] in ("indicator", "table"), "partly_relevant"),
    "finer_level": ("The search asks for the record's topic at a finer level than the record provides: a district, city or province when the record is national, or monthly data when it is annual, or individual records when it is aggregate.", lambda r: r["kind"] in ("indicator", "table", "study"), "partly_relevant"),
    "time_phrase": ("The search states its time period indirectly (e.g. 'since COVID', 'before the 2008 crisis', 'over the last decade', 'during the Ebola outbreak', a fiscal year like 'FY2019/20'). The period may or may not be covered by the record.", lambda r: r.get("year_start"), None),
    "negation": ("The search explicitly excludes something (e.g. 'excluding urban areas', 'not Nigeria', 'other than agriculture', 'without imputed values'), and the record is about what the search is asking otherwise.", None, None),
    "multi_ask": ("The search asks for two things at once (two topics, or two countries) and the record covers only one of them.", None, "partly_relevant"),
    "question": ("The search is a natural question asking for a fact or number that this record would let the user find (e.g. 'what share of children are vaccinated in Ghana?').", lambda r: r["kind"] in ("indicator", "table", "study"), "relevant"),
    "title_name": ("The search is the name of this survey, dataset, table or paper as a user would type it from memory: shortened, with the year or country, possibly an informal or partial name.", lambda r: r["kind"] in ("study", "dataset", "table", "document", "script"), "relevant"),
    "vague": ("The search is very short and vague: one or two generic words related to the record's broad area (e.g. 'health', 'data Kenya', 'agriculture stats').", None, None),
    "old_place": ("The search uses an older or alternative name of a country the record covers: {renamed}. The rest of the search matches the record's topic.", lambda r: any(c in RENAMED for c in r.get("countries") or []), "relevant"),
    "group_definition": ("The search names a population group in different terms than the record (e.g. 'women' when the record has female-headed households, 'teenagers' when it has youth aged 15-24, 'the elderly' when it has pensioners). The record covers a related but not identical group.", lambda r: r["kind"] in ("indicator", "table", "study"), "partly_relevant"),
    "lookalike": ("The search shares words with the record's title but means something different (e.g. 'bank erosion' for a banking indicator, 'capital city growth' for capital formation, 'labor during childbirth' for a labor market survey). The record does not answer it.", None, "irrelevant"),
}

PROMPT = """You write test searches for a development-data catalog search engine. A catalog record is shown below.

Write ONE search a real user might type, in {language}, for this test case:
{instruction}

Record ({label}):
{item}

Write it as a real person would type it in {language} (place names in their local spelling; acronyms may stay as they are).
Return only a JSON object: {{"query": "...", "expected": "relevant" | "partly_relevant" | "irrelevant", "note": "one short sentence on what the search tests"}}"""


def load_records(n: int, seed: int) -> list[dict]:
    rng = random.Random(seed)
    by_kind = defaultdict(dict)
    for f in sorted(Path("data/real").glob("c*/examples.jsonl"))[-30:]:
        for l in f.open():
            rec = json.loads(l)["record"]
            if len(rec.get("abstract") or rec.get("definition") or rec.get("notes") or "") > 80:
                by_kind[rec["kind"]][rec["id"]] = rec
    for p in Path("data/pools").glob("*.jsonl"):
        if p.name.endswith(("_fr.jsonl", "_es.jsonl")):
            continue
        for l in p.open():
            rec = json.loads(l)
            by_kind[rec["kind"]][rec["id"]] = rec
    share = {"indicator": 0.25, "study": 0.2, "document": 0.15, "dataset": 0.12, "table": 0.18, "geospatial": 0.05, "script": 0.05}
    out = []
    for k, s in share.items():
        pool = list(by_kind.get(k, {}).values())
        rng.shuffle(pool)
        out += pool[: int(n * s)]
    rng.shuffle(out)
    return out


def record_half(rec_id: str) -> int:
    """Stable half (0 or 1) of a record, to split one record pool between training searches and a held-out test."""
    import hashlib
    return hashlib.md5(rec_id.encode()).digest()[0] & 1


def cmd_write(a) -> None:
    rng = random.Random(a.seed)
    per = json.loads(a.per_phenomenon) if a.per_phenomenon else {}
    if a.records:  # one record file (e.g. a pool) instead of the default sources, optionally one half of it
        recs = [json.loads(l) for l in Path(a.records).read_text().splitlines()]
        recs = [r for r in recs if a.half is None or record_half(r["id"]) == a.half]
        rng.shuffle(recs)
    else:
        recs = load_records(max([a.per, *per.values()]) * 6, a.seed)
    only = set(a.phenomena.split(",")) if a.phenomena else None
    exclude = set()
    if a.exclude:  # records used by the held-out challenge suite
        exclude = {json.loads(l)["record"]["id"] for l in Path(a.exclude).read_text().splitlines()}
    jobs = []
    for name, (instr, filt, _) in PHENOMENA.items():
        if only and name not in only:
            continue
        cands = [r for r in recs if (filt is None or filt(r)) and r["id"] not in exclude]
        rng.shuffle(cands)
        for k, rec in enumerate(cands[: per.get(name, a.per)]):
            lang = rng.choices(list(LANG_W), weights=list(LANG_W.values()))[0]
            text = instr
            if name == "old_place":
                text = instr.format(renamed=", ".join(f"{RENAMED[c]} for {c}" for c in rec["countries"] if c in RENAMED))
            jobs.append({"id": f"{name}-{k}", "phenomenon": name, "language": lang, "record": rec, "instruction": text})

    def one(j):
        src = renderer(j["record"]["kind"])
        item = "\n".join(src.render(j["record"]))[:2500]
        for _ in range(3):
            model, urls = writer(j["id"]) if a.mix_writers else GEMMA
            t, _ = llm.chat(model, "You write realistic, varied test cases.", PROMPT.format(language=j["language"], instruction=j["instruction"],
                                                                                          label=src.label, item=item), temperature=0.9, max_tokens=300, base_urls=urls)
            js = llm.parse_json(t)
            q = str(js.get("query") or "").strip()
            if q and language_ok(q, j["language"]):
                return [{**j, "query": q, "writer_expected": js.get("expected"), "note": js.get("note")}]
        return []

    with ThreadPoolExecutor(a.workers) as ex:
        rows = [x for xs in ex.map(lambda j: G._safe(one, j), jobs) for x in xs]
    Path(a.out).mkdir(parents=True, exist_ok=True)
    (Path(a.out) / "searches.jsonl").write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows))
    print("written:", len(rows), "of", len(jobs), dict(Counter(r["phenomenon"] for r in rows)))


def cmd_label(a) -> None:
    from .signals import item_text, similarities
    rows = [json.loads(l) for l in (Path(a.out) / "searches.jsonl").read_text().splitlines()]
    prompt, version = {"v2": NARRATE_V2, "v3": NARRATE_V3}[a.prompt], a.render_version or RENDER_VERSION
    sims = similarities([r["query"] for r in rows], [item_text(r["record"]) for r in rows])
    u = Understander()
    for r, s in zip(rows, sims):
        p = {"query": r["query"], "language": r["language"], "record": r["record"], "facet_values": {}, "render_version": version}
        r["similarity"] = float(s)
        r["render_version"] = version
        r["prompt"] = a.prompt
        r["input_v2"] = render_v2(p, u, u.parse(r["query"], {}, level=version), float(s))

    def one(r):
        t = llm.parse_json(llm.chat(TEACHER[0], prompt, r["input_v2"], temperature=0.4, max_tokens=450, base_urls=TEACHER[1])[0])
        if t.get("verdict") not in ("relevant", "partly_relevant", "irrelevant"):
            return []
        try:
            g = llm.parse_json(llm.chat(GEMMA[0], prompt, r["input_v2"], temperature=0.0, max_tokens=450, base_urls=GEMMA[1])[0]).get("verdict")
        except Exception:
            g = None
        return [{**r, "verdict": t["verdict"], "narrative": str(t["narrative"]).strip(), "gemma4": g}]

    with ThreadPoolExecutor(a.workers) as ex:
        out = [x for xs in ex.map(lambda r: G._safe(one, r), rows) for x in xs]
    (Path(a.out) / a.labeled).write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in out))
    print("labeled:", len(out), "| Gemma 4 agrees with the teacher:", round(sum(r["gemma4"] == r["verdict"] for r in out) / max(1, len(out)), 3))


def cmd_build(a) -> None:
    rows = [json.loads(l) for l in (Path(a.out) / a.labeled).read_text().splitlines()]
    for name, strip in (("itemfirst", False), ("itemfirst_nosim", True)):
        d = Path(a.eval) / name
        d.mkdir(parents=True, exist_ok=True)
        out = []
        for r in rows:
            inp = item_first(r["input_v2"])
            if strip:
                inp = re.sub(r"\nRetrieval similarity: [-0-9.]+", "", inp)
            out.append({"input": inp, "output": f"{r['verdict']}\n{r['narrative']}", "verdict": r["verdict"], "language": r["language"],
                        "source": r["record"]["kind"], "labels": {"stored": r["verdict"], "gemma4": r["gemma4"]}, "majority": r["verdict"],
                        "unanimous": r["gemma4"] == r["verdict"], "style": r.get("writer_expected"), "mode": r["phenomenon"],
                        "relation": r["phenomenon"]})
        text = "".join(json.dumps(x, ensure_ascii=False) + "\n" for x in out)
        for split in ("test", "val"):
            (d / f"{split}.jsonl").write_text(text)
        (d / "train.jsonl").write_text(text.split("\n", 1)[0] + "\n")
    print("built", len(rows))
    tab = defaultdict(Counter)
    for r in rows:
        tab[r["phenomenon"]][r["verdict"]] += 1
    for k, c in tab.items():
        print(f"  {k:18s} teacher labels {dict(c)}")


def cmd_examples(a) -> None:
    """Training rows in the realistic example schema (for assemble_v2), only where the teacher and Gemma 4 agree."""
    from .understand import Understander
    rows = [json.loads(l) for l in (Path(a.out) / a.labeled).read_text().splitlines()]
    u = Understander()
    out = []
    for r in rows:
        if r.get("gemma4") != r["verdict"]:
            continue
        out.append({"query": r["query"], "language": r["language"], "record": r["record"], "facet_values": {},
                    "parsed": u.parse(r["query"], {}, level=r.get("render_version", 2)), "similarity": r["similarity"],
                    "render_version": r.get("render_version", 2),
                    "verdict": r["verdict"], "narrative": r["narrative"], "topic": f"phen-{r['phenomenon']}-{r['record']['id']}",
                    "mode": "anchored", "relation": r["phenomenon"], "wording": r["phenomenon"], "style": None,
                    "writer": "mixed"})
    (Path(a.out) / a.examples).write_text("".join(json.dumps(x, ensure_ascii=False) + "\n" for x in out))
    print(f"training rows: {len(out)} of {len(rows)} (teacher and Gemma 4 agree)", dict(Counter(x["relation"] for x in out)))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("stage", choices=["write", "label", "build", "examples"])
    ap.add_argument("--out", default="data/challenge")
    ap.add_argument("--eval", default="data/eval_challenge")
    ap.add_argument("--per", type=int, default=150)
    ap.add_argument("--per-phenomenon", default=None, help='JSON {"synonym": 1500, ...} overriding --per')
    ap.add_argument("--exclude", default=None, help="searches.jsonl whose records are not used (the held-out suite)")
    ap.add_argument("--mix-writers", action="store_true", help="the training writers (data/real/writers.json) instead of Gemma 4 only")
    ap.add_argument("--records", default=None, help="JSONL of records to write searches for (default: realistic chunks and pools)")
    ap.add_argument("--half", type=int, choices=[0, 1], default=None, help="with --records: only records in this half (record_half)")
    ap.add_argument("--phenomena", default=None, help="comma-separated subset of PHENOMENA")
    ap.add_argument("--prompt", choices=["v2", "v3"], default="v2", help="teacher prompt: NARRATE_V2, or NARRATE_V3 with the verdict rules")
    ap.add_argument("--render-version", type=int, default=None, help="render version of the inputs (default RENDER_VERSION)")
    ap.add_argument("--labeled", default="labeled.jsonl", help="labelled file name inside --out")
    ap.add_argument("--examples", default="examples.jsonl", help="training rows file name inside --out")
    ap.add_argument("--seed", type=int, default=11)
    ap.add_argument("--workers", type=int, default=16)
    a = ap.parse_args()
    {"write": cmd_write, "label": cmd_label, "build": cmd_build, "examples": cmd_examples}[a.stage](a)


if __name__ == "__main__":
    main()
