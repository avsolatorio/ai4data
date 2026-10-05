"""A test set built from metadata-schemas records (NADA catalogs): searches written against each record for every
relation, inputs rendered through relevance_explainer.schemas, labels from the teacher and Gemma 4.

  python -m relevance_explainer.synth.schemas_eval searches --src data/nada_demo.jsonl --out data/schemas_eval
  python -m relevance_explainer.synth.schemas_eval label --out data/schemas_eval
  python -m relevance_explainer.synth.schemas_eval build --out data/schemas_eval      # writes <out>/itemfirst/test.jsonl
"""
from __future__ import annotations

import argparse
import json
import random
import re
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from . import generate as G
from . import llm
from .realistic import (ANCHOR_PROMPT, GROUPS, RELATION, RELATION_TEXT, STYLE, SURFACE_TEXT, Specs, item_first,
                        language_ok, must_lines, recase, writer)
from .understand import GROUP_EVIDENCE, _record_iso3, _record_text
from ..schemas import RENDERERS, TRAINED_KINDS, similarity, to_input, to_record

GEMMA = ["http://localhost:8002/v1"]


def spec_for(rel: str, rec: dict, sp: Specs, r: random.Random) -> dict:
    """The search constraints for one relation, as in realistic.cmd_anchored."""
    s = sp.base()
    covered = sorted(_record_iso3(rec))
    yrs = (rec["year_start"], rec["year_end"]) if rec.get("year_start") and rec.get("year_end") else None
    dated = yrs and rec["kind"] not in ("document",)
    s.update(relation=rel, geo=None, time=None)
    if rel == "full":
        s["geo"] = sp.geography(r.choice(["none", "country", "country"]), allowed=covered[:40]) if 0 < len(covered) <= 60 else None
        s["time"] = sp.time(r.choice(["none", "year", "range"]), inside=yrs) if dated else None
        s["group"] = s["group"] if s.get("group") and re.search(GROUP_EVIDENCE[s["group"]], _record_text(rec), re.I) else None
    elif rel == "wrong_place":
        s["geo"] = sp.geography("country", avoid=set(covered))
        s["group"] = None
    elif rel == "wrong_time":
        s["time"] = sp.time("year", outside=yrs) if dated else sp.time("year")
        s["group"] = None
    elif rel == "wrong_group":
        missing = [g for g in GROUPS if not re.search(GROUP_EVIDENCE[g], _record_text(rec), re.I)]
        s["group"] = r.choice(missing) if missing else r.choice(GROUPS)
    elif rel in ("related", "other_topic"):
        s["geo"] = sp.geography("country", allowed=covered[:40]) if covered and r.random() < 0.6 else None
        s["group"] = None
    else:  # unrelated
        s["topic"] = r.choice(["urban air quality", "fisheries subsidies", "remittance costs", "tuberculosis treatment",
                               "electricity tariffs", "teacher absenteeism", "crop insurance", "port congestion"])
        s["geo"] = sp.geography("country")
        s["group"] = None
    s["datatype"] = "none"
    return s


def cmd_searches(a) -> None:
    rows = [json.loads(l) for l in Path(a.src).read_text().splitlines()]
    geo = json.loads(Path(a.geo).read_text())
    sp = Specs(geo, a.seed)
    jobs = []
    for k, row in enumerate(rows):
        rec = to_record(row["detail"], row["row"])
        for rel in RELATION:
            r = random.Random(f"{a.seed}-{k}-{rel}")
            jobs.append((f"s{k}-{rel}", rec, spec_for(rel, rec, sp, r)))

    def one(job):
        sid, rec, s = job
        r = random.Random(sid)
        item = "\n".join(RENDERERS[rec["kind"]].render(rec))
        must = must_lines(s)
        rel = s["relation"]
        for _ in range(3):
            wm, wurls = writer(sid)
            text, _ = llm.chat(wm, G.QUERY_SYSTEM, ANCHOR_PROMPT.format(
                relation=RELATION_TEXT[rel] if rel != "unrelated" else f"about {s['topic']}, a topic this record does not cover",
                label=RENDERERS[rec["kind"]].label, item=item[:2500], language=s["language"], style=STYLE[s["style"]][1],
                surface=SURFACE_TEXT[s["surface"]],
                must=("The search must express:\n" + must) if (s.get("geo") or s.get("time") or s.get("group")) else "Do not mention a place, year or population group unless the relation requires it."),
                temperature=0.9, max_tokens=400, base_urls=wurls)
            j = llm.parse_json(text)
            q = str(j.get("query") or "").strip()
            if q and language_ok(q, s["language"]):
                return [{"id": sid, "writer": wm, "query": recase(q, r), "language": s["language"], "style": s["style"],
                         "relation": rel, "record": rec, "schema": rec["schema"]}]
        return []

    with ThreadPoolExecutor(a.workers) as ex:
        out = [x for xs in ex.map(lambda j: G._safe(one, j), jobs) for x in xs]
    for x in out:
        x["similarity"] = round(similarity(x["query"], x["record"]), 4)
    Path(a.out).mkdir(parents=True, exist_ok=True)
    (Path(a.out) / "searches.jsonl").write_text("".join(json.dumps(x, ensure_ascii=False) + "\n" for x in out))
    print(f"searches: {len(out)} of {len(jobs)}")


def cmd_label(a) -> None:
    from .realistic import NARRATE_V2
    rows = [json.loads(l) for l in (Path(a.out) / "searches.jsonl").read_text().splitlines()]

    def one(x):
        inp = to_input(x["record"], x["query"], x["language"], sim=x["similarity"], fmt="full")
        text, _ = llm.chat(G.NARRATE_MODEL, NARRATE_V2, inp, temperature=0.4, max_tokens=450)
        j = llm.parse_json(text)
        if j.get("verdict") not in ("relevant", "partly_relevant", "irrelevant"):
            return []
        try:
            g = llm.parse_json(llm.chat("gemma-4-31b", NARRATE_V2, inp, temperature=0.0, max_tokens=450, base_urls=GEMMA)[0]).get("verdict")
        except Exception:
            g = None
        return [{**x, "input_v2": inp, "verdict": j["verdict"], "narrative": str(j["narrative"]).strip(), "gemma4": g}]

    with ThreadPoolExecutor(a.workers) as ex:
        out = [y for ys in ex.map(lambda x: G._safe(one, x), rows) for y in ys]
    (Path(a.out) / "labeled.jsonl").write_text("".join(json.dumps(x, ensure_ascii=False) + "\n" for x in out))
    agree = sum(x["gemma4"] == x["verdict"] for x in out) / max(len(out), 1)
    print(f"labeled: {len(out)} of {len(rows)}; Gemma 4 agrees with the teacher on {agree:.3f}")


def cmd_build(a) -> None:
    rows = [json.loads(l) for l in (Path(a.out) / "labeled.jsonl").read_text().splitlines()]
    d = Path(a.out) / "itemfirst"
    d.mkdir(parents=True, exist_ok=True)
    out = [{"input": item_first(x["input_v2"]), "output": f"{x['verdict']}\n{x['narrative']}", "verdict": x["verdict"],
            "language": x["language"], "source": x["record"]["kind"], "schema": x["schema"],
            "trained_kind": x["record"]["kind"] in TRAINED_KINDS, "labels": {"stored": x["verdict"], "gemma4": x["gemma4"]},
            "majority": x["verdict"], "unanimous": x["gemma4"] == x["verdict"], "style": x["style"], "mode": "anchored",
            "relation": x["relation"]} for x in rows]
    text = "".join(json.dumps(x, ensure_ascii=False) + "\n" for x in out)
    for split in ("test", "val"):
        (d / f"{split}.jsonl").write_text(text)
    (d / "train.jsonl").write_text(text.split("\n", 1)[0] + "\n")
    print("built", len(out))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("stage", choices=["searches", "label", "build"])
    ap.add_argument("--src", default="data/nada_demo.jsonl")
    ap.add_argument("--out", default="data/schemas_eval")
    ap.add_argument("--geo", default="data/geo/geo.json")
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--workers", type=int, default=32)
    a = ap.parse_args()
    {"searches": cmd_searches, "label": cmd_label, "build": cmd_build}[a.stage](a)


if __name__ == "__main__":
    main()
