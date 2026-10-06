"""Record pools for item types the catalog APIs used so far do not provide. Each pool is a JSONL file of records in the
shape of `relevance_explainer.schemas` (kind table, geospatial or script), searched locally by `sources.PoolSource`.

  python -m relevance_explainer.synth.pools statcan --out data/pools/table_statcan.jsonl --n 3000
  python -m relevance_explainer.synth.pools ilostat --out data/pools/table_ilostat.jsonl
  python -m relevance_explainer.synth.pools hdxgeo  --out data/pools/geospatial_hdx.jsonl --n 3000
  python -m relevance_explainer.synth.pools nada    --out data/pools/script_rrr.jsonl --base https://reproducibility.worldbank.org

Sources: Statistics Canada Web Data Service (tables, English and French metadata), ILOSTAT table of contents (tables,
English, French and Spanish), HDX datasets with SHP or GeoJSON resources (map layers), NADA catalogs (any schema type).
"""
from __future__ import annotations

import argparse
import csv
import io
import json
import random
import re
import time
import urllib.parse
import urllib.request
from pathlib import Path

UA = {"User-Agent": "relevance-explainer research (record pools)"}


def _get(url: str, post=None, timeout: int = 90):
    data = json.dumps(post).encode() if post is not None else None
    req = urllib.request.Request(url, data, {**UA, **({"Content-Type": "application/json"} if data else {})})
    for attempt in range(4):
        try:
            with urllib.request.urlopen(req, timeout=timeout) as r:
                return r.read()
        except Exception:
            if attempt == 3:
                raise
            time.sleep(5 * (attempt + 1))


def _clip(s, n: int) -> str:
    s = re.sub(r"\s+", " ", str(s or "")).strip()
    return s if len(s) <= n else s[: n - 1].rstrip() + "…"


def _year(s):
    m = re.search(r"(1[89]|20)\d{2}", str(s or ""))
    return int(m.group(0)) if m else None


def _write(rows: list[dict], out: str) -> None:
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    Path(out).write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows))
    print(f"{len(rows)} records -> {out}")


# ---------- Statistics Canada ----------
STATCAN = "https://www150.statcan.gc.ca/t1/wds/rest"


def _statcan_table(o: dict, codes: dict, lang: str) -> dict:
    L = "En" if lang == "en" else "Fr"
    dims = o.get("dimension") or []
    cols = [f"{d['dimensionName' + L]} ({len(d.get('member') or [])})" for d in dims]
    geo = next((d for d in dims if d.get("dimensionNameEn", "").startswith("Geography")), None)
    places = [m["memberName" + L] for m in (geo or {}).get("member", [])]
    levels = sorted({m.get("geoLevel") for m in (geo or {}).get("member", []) if m.get("geoLevel") is not None})
    gran = {0: "country", 2: "province or territory", 3: "region", 4: "census division", 5: "census subdivision",
            6: "census metropolitan area", 7: "census agglomeration"}
    note = " ".join(f.get("footnotes" + L) or "" for f in (o.get("footnote") or [])[:2])
    pid = str(o["productId"])
    return {"kind": "table", "id": f"statcan-{pid}", "name": o["cubeTitle" + L],
            "table_number": f"{pid[:2]}-{pid[2:4]}-{pid[4:8]}-01", "producer": "Statistics Canada" if lang == "en" else "Statistique Canada",
            "date": str(o.get("releaseTime") or "")[:10], "columns": _clip(", ".join(cols), 300),
            "rows": _clip(f"{codes['frequency'].get(o.get('frequencyCode'), {}).get(L, '')} periods {str(o.get('cubeStartDate'))[:7]} to {str(o.get('cubeEndDate'))[:7]}", 120),
            "granularity": ", ".join(gran.get(l, f"level {l}") for l in levels) or None,
            "geography": _clip(", ".join(places[:14]) + (f" and {len(places) - 14} more" if len(places) > 14 else ""), 220) or None,
            "sources": _clip("; ".join(codes["survey"].get(s, {}).get(L, "") for s in o.get("surveyCode") or []), 160) or None,
            "topics": [codes["subject"].get(s, {}).get(L) for s in o.get("subjectCode") or [] if codes["subject"].get(s)][:4],
            "tags": [], "abstract": _clip(note, 450), "unit_of_analysis": None, "universe": None,
            "countries": ["Canada"], "countries_truncated": False, "frequency": codes["frequency"].get(o.get("frequencyCode"), {}).get("En"),
            "year_start": _year(o.get("cubeStartDate")), "year_end": _year(o.get("cubeEndDate")), "year_label": "years",
            "schema": "table", "meta_language": "English" if lang == "en" else "French"}


def build_statcan(a) -> None:
    cs = json.loads(_get(f"{STATCAN}/getCodeSets"))["object"]
    codes = {"subject": {}, "survey": {}, "frequency": {}}
    for s in cs["subject"]:
        codes["subject"][s["subjectCode"]] = {"En": s["subjectEn"], "Fr": s["subjectFr"]}
    for s in cs["survey"]:
        codes["survey"][s["surveyCode"]] = {"En": s["surveyEn"], "Fr": s["surveyFr"]}
    for s in cs["frequency"]:
        codes["frequency"][s["frequencyCode"]] = {"En": s["frequencyDescEn"], "Fr": s["frequencyDescFr"]}
    cubes = [c for c in json.loads(_get(f"{STATCAN}/getAllCubesListLite")) if str(c.get("archived")) != "1"]
    random.Random(0).shuffle(cubes)
    cubes = cubes[: a.n]
    en, fr = [], []
    for i in range(0, len(cubes), 50):
        batch = [{"productId": c["productId"]} for c in cubes[i : i + 50]]
        try:
            res = json.loads(_get(f"{STATCAN}/getCubeMetadata", post=batch, timeout=180))
        except Exception as e:
            print("skip batch", i, e)
            continue
        for r in res:
            if r.get("status") != "SUCCESS":
                continue
            o = r["object"]
            en.append(_statcan_table(o, codes, "en"))
            fr.append(_statcan_table(o, codes, "fr"))
        time.sleep(0.5)
    _write(en, a.out)
    _write(fr, a.out.replace(".jsonl", "_fr.jsonl"))


# ---------- ILOSTAT ----------
def build_ilostat(a) -> None:
    for lang in ("en", "fr", "es"):
        text = _get(f"https://rplumber.ilo.org/metadata/toc/indicator/?lang={lang}&format=.csv").decode("utf-8-sig")
        rows = list(csv.DictReader(io.StringIO(text)))
        out = []
        for r in rows:
            n = int(r.get("n.ref_area") or 0)
            classif = r.get("classif.labels") or ""
            out.append({"kind": "table", "id": f"ilostat-{r['id']}", "name": r["indicator.label"],
                        "table_number": r["id"], "producer": {"en": "International Labour Organization (ILOSTAT)",
                                                              "fr": "Organisation internationale du Travail (ILOSTAT)",
                                                              "es": "Organización Internacional del Trabajo (ILOSTAT)"}[lang],
                        "date": str(r.get("last.update") or "")[:10],
                        "columns": _clip(", ".join(["ref_area", "time", *[c.strip() for c in classif.split(",") if c.strip()], "obs_value"]), 200),
                        "rows": f"{n} reference areas x {r.get('freq.label')} periods",
                        "granularity": "country" + (", regions" if r.get("with.region") in ("TRUE", "True", "1") else ""),
                        "geography": f"{n} countries and territories", "sources": r.get("database.label") or None,
                        "topics": [r.get("subject.label")] if r.get("subject.label") else [], "tags": [],
                        "abstract": _clip(f"{r.get('rep_var.label') or ''}. {r.get('database.label') or ''}", 300),
                        "unit_of_analysis": None, "universe": None,
                        "countries": [], "countries_truncated": n > 0, "frequency": r.get("freq.label"),
                        "year_start": _year(r.get("data.start")), "year_end": _year(r.get("data.end")), "year_label": "years",
                        "schema": "table", "meta_language": {"en": "English", "fr": "French", "es": "Spanish"}[lang]})
        _write(out, a.out if lang == "en" else a.out.replace(".jsonl", f"_{lang}.jsonl"))


# ---------- HDX map layers ----------
def build_hdxgeo(a) -> None:
    from .sources import HDX
    out, start = [], 0
    while len(out) < a.n:
        qs = urllib.parse.urlencode({"fq": "res_format:SHP OR res_format:GeoJSON OR res_format:\"Geodatabase\"", "rows": 500, "start": start,
                                     "sort": "metadata_modified desc"})
        res = json.loads(_get(f"https://data.humdata.org/api/3/action/package_search?{qs}"))["result"]["results"]
        if not res:
            break
        start += 500
        for h in res:
            rec = HDX().hydrate(h, "")
            if not rec:
                continue
            fmts = sorted({str(r.get("format") or "").upper() for r in h.get("resources") or [] if r.get("format")})
            out.append({"kind": "geospatial", "id": "hdxgeo-" + rec["id"], "name": rec["name"], "producer": rec["organization"],
                        "representation": "vector" if any(f in fmts for f in ("SHP", "GEOJSON", "KML", "KMZ", "GEODATABASE", "GEOPACKAGE")) else None,
                        "formats": fmts[:5], "frequency": rec["update_frequency"],
                        "geography": _clip(h.get("dataset_source"), 80) or None, "topics": [], "tags": rec["tags"],
                        "abstract": rec["notes"], "purpose": None, "countries": rec["countries"],
                        "countries_truncated": False, "year_start": rec["year_start"], "year_end": rec["year_end"],
                        "year_label": "years", "schema": "geospatial", "meta_language": "English"})
        time.sleep(1)
    _write(out[: a.n], a.out)


# ---------- NADA catalogs (any schema type) ----------
def build_nada(a) -> None:
    from ..schemas import to_record
    api = a.base.rstrip("/") + "/index.php/api/catalog"
    rows, page = [], 1
    while True:
        res = json.loads(_get(f"{api}/search?format=json&ps=100&page={page}"))["result"]
        rows += res["rows"]
        if not res["rows"] or len(rows) >= int(res["found"]):
            break
        page += 1
    out = []
    for r in rows[: a.n]:
        try:
            d = json.loads(_get(f"{api}/{urllib.parse.quote(r['idno'])}"))
            rec = to_record(d, r)
        except Exception as e:
            print("skip", r.get("idno"), e)
            continue
        if rec.get("name"):
            out.append({**rec, "id": f"nada-{rec.get('id') or r['idno']}", "meta_language": "English"})
        time.sleep(0.2)
    _write(out, a.out)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("source", choices=["statcan", "ilostat", "hdxgeo", "nada"])
    ap.add_argument("--out", required=True)
    ap.add_argument("--n", type=int, default=3000)
    ap.add_argument("--base", default="https://reproducibility.worldbank.org")
    a = ap.parse_args()
    {"statcan": build_statcan, "ilostat": build_ilostat, "hdxgeo": build_hdxgeo, "nada": build_nada}[a.source](a)


if __name__ == "__main__":
    main()
