"""Test set of records from catalogs absent from training (evalsets/unseen_catalogs; notebook 2026-10-16).

  python scripts/unseen_catalogs.py fetch --out evalsets/unseen_catalogs/records.jsonl     # rebuild the records
  python scripts/unseen_catalogs.py run --bundle dist/relevance-explainer-c-dpo --out answers.jsonl [--render-version 5]
                                        [--kind opendata|dataset]
  python scripts/unseen_catalogs.py report answers.jsonl [answers_b.jsonl ...]

fetch reads the records by their fixed identifiers from Our World in Data (charts, as indicator series), open.canada.ca
and data.gov.au (CKAN, through schemas.ckan_record) and Zenodo (datasets, as opendata records). The committed
records.jsonl is the snapshot of 2026-10-16; fetch writes a new file so a later snapshot can be compared with it.
run explains every case of cases.tsv with a model bundle; --kind dataset renders the opendata records with the HDX
renderer, the only dataset type before render version 5. report prints verdict agreement with the labels (strict, and
lenient when a case accepts a second label), trust states and faithfulness bands, by source and language group.
"""
from __future__ import annotations

import argparse
import csv
import io
import json
import re
import sys
import urllib.parse
import urllib.request
from collections import Counter
from pathlib import Path

sys.path.insert(0, "src")
SET = Path("evalsets/unseen_catalogs")
UA = {"User-Agent": "curl/8.5.0"}
CKAN = {"canada": ("https://open.canada.ca/data/api/3/action/package_show", ["Canada"]),
        "australia": ("https://data.gov.au/data/api/3/action/package_show", ["Australia"])}


def _get(url: str) -> bytes:
    with urllib.request.urlopen(urllib.request.Request(url, headers=UA), timeout=90) as r:
        return r.read()


def owid(slug: str) -> dict:
    from relevance_explainer.schemas import _clip, split_places
    md = json.loads(_get(f"https://ourworldindata.org/grapher/{slug}.metadata.json"))
    col = next(iter(md["columns"].values()))
    rows = list(csv.DictReader(io.StringIO(_get(f"https://ourworldindata.org/grapher/{slug}.csv?useColumnShortNames=true").decode())))
    countries, _ = split_places(sorted({r["entity"] for r in rows}))
    years = [int(r["year"]) for r in rows if r["year"].lstrip("-").isdigit()]
    return {"kind": "indicator", "id": f"owid:{slug}", "name": md["chart"]["title"], "database": "Our World in Data",
            "source": _clip(col.get("citationShort") or md["chart"].get("citation"), 160), "unit": col.get("unit"),
            "frequency": "Annual", "definition": _clip(col.get("descriptionShort") or md["chart"].get("subtitle"), 450),
            "concept": None, "limitation": None, "topics": [], "year_start": min(years) if years else None,
            "year_end": max(years) if years else None, "countries": countries, "year_label": "years"}


def zenodo(rid: str) -> dict:
    from relevance_explainer.schemas import _clip, countries_in_text
    h = json.loads(_get(f"https://zenodo.org/api/records/{rid}"))
    m = h["metadata"]
    desc = re.sub(r"<[^>]+>|&nbsp;", " ", m.get("description") or "")
    y = int(m["publication_date"][:4]) if m.get("publication_date") else None
    return {"kind": "opendata", "id": f"zenodo:{rid}", "name": m.get("title"), "organization": "Zenodo",
            "source": _clip("; ".join(c["name"] for c in m.get("creators") or []), 120), "notes": _clip(desc, 500),
            "tags": (m.get("keywords") or [])[:8],
            "formats": sorted({f["key"].rsplit(".", 1)[-1].upper() for f in h.get("files") or [] if "." in f["key"]})[:5],
            "countries": countries_in_text(m.get("title") or "", desc), "year_start": y, "year_end": y,
            "year_label": "publication year", "frequency": None, "update_frequency": None}


def fetch(out: Path) -> None:
    from relevance_explainer.schemas import ckan_record
    ids = list(dict.fromkeys(c["record"] for c in cases()))
    recs = []
    for rid in ids:
        src, key = rid.split(":", 1)
        try:
            if src == "owid":
                rec = owid(key)
            elif src == "zenodo":
                rec = zenodo(key)
            else:
                url, countries = CKAN[src]
                rec = {**ckan_record(json.loads(_get(f"{url}?id={urllib.parse.quote(key)}"))["result"], countries), "id": rid}
        except Exception as e:  # noqa: BLE001  one unreachable portal should not lose the others
            print(f"{rid}: {e!r}", file=sys.stderr)
            continue
        recs.append(rec)
    out.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in recs))
    print(f"{len(recs)} of {len(ids)} records -> {out}")


def cases() -> list[dict]:
    return list(csv.DictReader(open(SET / "cases.tsv", encoding="utf-8"), delimiter="\t"))


def run(bundle: str, out: Path, version: int, kind: str) -> None:
    from relevance_explainer import inference
    recs = {r["id"]: r for r in map(json.loads, open(SET / "records.jsonl", encoding="utf-8"))}
    ex = inference.Explainer(bundle, threads=8)
    with open(out, "w") as f:
        for c in cases():
            rec = dict(recs[c["record"]])
            if rec["kind"] == "opendata" and kind == "dataset":
                rec["kind"] = "dataset"
            text = inference.render(rec, c["search"], c["language"], render_version=version, understander=ex.understander)
            o = ex.explain_text(text)
            f.write(json.dumps({**c, **{k: o[k] for k in ("verdict", "verdict_set", "trust", "verdict_view", "faithfulness",
                                                          "all_verified", "tldr", "explanation")}, "input": text},
                               ensure_ascii=False) + "\n")
    print(out)


def report(paths: list[str]) -> None:
    for p in paths:
        rows = [json.loads(l) for l in open(p)]
        print(p)
        for group in ("supported", "unsupported"):
            rs = [r for r in rows if (r["group"] == "unsupported") == (group == "unsupported")]
            if not rs:
                continue
            strict = sum(r["verdict"] == r["label"] for r in rs)
            lenient = sum(r["verdict"] in (r["label"], r["also_ok"]) for r in rs)
            print(f"  {group}: {len(rs)} cases, verdict strict {strict} ({strict / len(rs):.0%}), lenient {lenient} "
                  f"({lenient / len(rs):.0%})")
            print("    trust", dict(Counter(r["trust"]["state"] for r in rs)), "faith bands",
                  dict(Counter(r["faithfulness"]["band"] if r["faithfulness"] else "none" for r in rs)))
            if group == "supported":
                by, tot = Counter(), Counter()
                for r in rs:
                    s = r["record"].split(":")[0]
                    tot[s] += 1
                    by[s] += r["verdict"] in (r["label"], r["also_ok"])
                print("    lenient by source", {k: f"{by[k]}/{tot[k]}" for k in tot})
                wrong_checked = sum(r["trust"]["state"] == "checked" and r["verdict"] not in (r["label"], r["also_ok"]) for r in rs)
                print(f"    wrong verdicts shown as checked: {wrong_checked}")


def main() -> None:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    f = sub.add_parser("fetch")
    f.add_argument("--out", required=True)
    r = sub.add_parser("run")
    r.add_argument("--bundle", required=True)
    r.add_argument("--out", required=True)
    r.add_argument("--render-version", type=int, default=None)
    r.add_argument("--kind", choices=["opendata", "dataset"], default="opendata")
    p = sub.add_parser("report")
    p.add_argument("answers", nargs="+")
    a = ap.parse_args()
    if a.cmd == "fetch":
        fetch(Path(a.out))
    elif a.cmd == "run":
        from relevance_explainer.inference import RENDER_VERSION
        run(a.bundle, Path(a.out), a.render_version or RENDER_VERSION, a.kind)
    else:
        report(a.answers)


if __name__ == "__main__":
    main()
