"""Turn metadata written in the World Bank metadata schemas (github.com/worldbank/metadata-schemas) into model inputs.

Supported schema types: microdata (DDI), timeseries, document, table, geospatial (ISO 19139), image (IPTC or DCMI), video,
script (research projects) and timeseries-db. The four types the model was trained on (survey microdata, indicator series,
documents, HDX datasets) map onto the training record and rendering exactly; the other types get their own item lines
in the same "Field: value" style, with countries, years and topics normalised so the query checks work the same way.

  from relevance_explainer.schemas import to_record, to_input
  rec = to_record(metadata)                       # a NADA catalog entry, or the bare schema object
  text = to_input(rec, "pobreza rural Senegal 2020", language="Spanish")
"""
from __future__ import annotations

import re
from functools import lru_cache

from .synth.sources import SOURCES, Documents, _coverage

# NADA's "type" values and the schema each one uses
SCHEMA_TYPES = ("microdata", "timeseries", "document", "table", "geospatial", "image", "video", "script", "timeseriesdb")
_TOP_KEYS = {"study_desc": "microdata", "series_description": "timeseries", "document_description": "document",
             "table_description": "table", "image_description": "image", "video_description": "video",
             "project_desc": "script", "database_description": "timeseriesdb"}
_ALIASES = {"survey": "microdata", "ddi": "microdata", "indicator": "timeseries", "timeseries-db": "timeseriesdb",
            "database": "timeseriesdb", "photo": "image", "research_project": "script"}


def _clip(s, n: int) -> str:
    s = re.sub(r"\s+", " ", str(s or "")).strip()
    return s if len(s) <= n else s[: n - 1].rstrip() + "…"


def _year(s) -> int | None:
    m = re.search(r"(1[89]|20)\d{2}", str(s or ""))
    return int(m.group(0)) if m else None


def _names(items, key: str = "name") -> list[str]:
    """Names from a list of {name: ...} objects or plain strings."""
    out = []
    for x in items or []:
        v = x.get(key) if isinstance(x, dict) else x
        if v and str(v).strip():
            out.append(str(v).strip())
    return list(dict.fromkeys(out))


def _join(items, n: int, key: str = "name") -> str:
    return _clip(", ".join(_names(items, key)), n)


@lru_cache(maxsize=1)
def _geo():
    import json
    from .synth.understand import GEO, _country_index, norm
    countries = json.loads(GEO.read_text())["countries"]
    return _country_index(), norm, {iso3: c["name"] for iso3, c in countries.items()}


def split_places(units) -> tuple[list[str], list[str]]:
    """Separate a geographic list into countries (canonical names) and other places (regions, aggregates, 'World').
    Accepts {name, code} objects or strings; ISO3 codes are trusted when the name is not recognised."""
    idx, norm, canon = _geo()
    countries, other = [], []
    for u in units or []:
        name = (u.get("name") if isinstance(u, dict) else u) or ""
        code = (u.get("code") or u.get("abbreviation") or "").upper() if isinstance(u, dict) else ""
        iso3 = idx.get(norm(name)) or (code if code in canon else None)
        if iso3:
            countries.append(canon[iso3])
        elif name.strip():
            other.append(name.strip())
    return list(dict.fromkeys(countries)), list(dict.fromkeys(other))


def countries_in_text(*texts: str) -> list[str]:
    """Countries named in free text (titles, place keywords), matched on whole words."""
    idx, norm, canon = _geo()
    found = []
    for t in texts:
        nt = " " + re.sub(r"[^\w']+", " ", norm(t or "")) + " "
        for name, iso3 in idx.items():
            key = " " + re.sub(r"[^\w]+", " ", name).strip() + " "
            if len(name) > 3 and key in nt:
                found.append(canon[iso3])
    return list(dict.fromkeys(found))


def _span(periods, start: str = "start", end: str = "end") -> tuple[int | None, int | None]:
    ys = [y for p in periods or [] for y in (_year(p.get(start)), _year(p.get(end))) if y]
    return (min(ys), max(ys)) if ys else (None, None)


def detect(md: dict) -> tuple[str, dict]:
    """Return (schema type, metadata object) for a NADA catalog entry, an API response or a bare schema object."""
    md = md.get("dataset", md) if isinstance(md, dict) else md
    st = md.get("schematype") or md.get("type")
    inner = md.get("metadata") if isinstance(md.get("metadata"), dict) else md
    st = _ALIASES.get(str(st or "").lower(), str(st or "").lower())
    if st not in SCHEMA_TYPES:
        st = next((t for k, t in _TOP_KEYS.items() if k in inner), None)
        if st is None and "description" in inner and "identificationInfo" in (inner.get("description") or {}):
            st = "geospatial"
    if st is None:
        raise ValueError("unrecognised metadata: expected one of the metadata-schemas types")
    return st, inner


def to_record(md: dict, row: dict | None = None) -> dict:
    """Normalise schema metadata to the record the model's renderers use. `row` is an optional NADA search row,
    used as a fallback for countries and years."""
    st, m = detect(md)
    row = row or (md.get("dataset", md) if isinstance(md, dict) and "metadata" in md.get("dataset", md) else {})
    rec = _BUILDERS[st](m)
    rec.setdefault("id", rec.get("id") or row.get("idno"))
    if not rec.get("countries") and row.get("nation"):
        rec["countries"], extra = split_places([c.strip() for c in str(row["nation"]).split(",")])
        rec["countries_truncated"] = "..." in str(row["nation"])
    if rec["kind"] == "study" and rec.get("varcount") is None and str(row.get("varcount") or "").isdigit():
        rec["varcount"] = int(row["varcount"])
    if not rec.get("year_start") and row.get("year_start"):
        rec["year_start"], rec["year_end"] = _year(row["year_start"]), _year(row.get("year_end") or row["year_start"])
    rec.setdefault("countries", [])
    rec.setdefault("countries_truncated", False)
    rec.setdefault("frequency", None)
    rec.setdefault("year_start", None)
    rec.setdefault("year_end", rec["year_start"])
    rec["schema"] = st
    return rec


# ---------- the four trained types: same fields as the training records ----------
def _microdata(m: dict) -> dict:
    sd = m.get("study_desc") or {}
    info, method = sd.get("study_info") or {}, sd.get("method") or {}
    ts = sd.get("title_statement") or {}
    sub = ts.get("sub_title")
    countries, other = split_places(info.get("nation"))
    coll = (info.get("coll_dates") or [{}])
    ys, ye = _span(coll)
    if ys is None:
        ys, ye = _span(info.get("time_periods"))
    mode = (method.get("data_collection") or {}).get("coll_mode") or []
    geog = info.get("geog_coverage") or ", ".join(other)
    nvar = sum(int(f.get("var_count") or 0) for f in (m.get("data_files") or []) if str(f.get("var_count") or "").isdigit()) or None
    return {"kind": "study", "id": ts.get("idno"), "name": f"{ts.get('title') or ''}{' (' + sub + ')' if sub else ''}",
            "producer": _clip(_join(sd.get("authoring_entity"), 200), 100),
            "data_kind": _clip(info.get("data_kind"), 60), "unit_of_analysis": _clip(info.get("analysis_unit"), 80),
            "universe": _clip(info.get("universe"), 200), "geography": _clip(geog, 120),
            "collection_mode": _clip(", ".join(mode) if isinstance(mode, list) else mode, 80),
            "topics": _names(info.get("topics"), "topic")[:6], "abstract": _clip(info.get("abstract"), 500),
            "varcount": nvar, "countries": countries, "countries_truncated": False,
            "year_start": ys, "year_end": ye, "frequency": None, "year_label": "survey years"}


def _timeseries(m: dict) -> dict:
    sd = m.get("series_description") or {}
    countries, other = split_places((sd.get("ref_country") or []) + (sd.get("geographic_units") or []))
    ys, ye = _span(sd.get("time_periods"))
    db = sd.get("database_name") or sd.get("database_id") or _join(sd.get("databases"), 100)
    return {"kind": "indicator", "id": sd.get("idno"), "name": sd.get("name") or sd.get("display_name") or "",
            "database": db or None, "source": next(iter(_names(sd.get("authoring_entity"))), None) or _join(sd.get("sources"), 100, "name") or None,
            "unit": sd.get("measurement_unit") or None, "frequency": sd.get("periodicity") or None,
            "definition": _clip(sd.get("definition_short") or sd.get("definition_long"), 450),
            "concept": _clip(sd.get("statistical_concept") or sd.get("methodology"), 250),
            "limitation": _clip(sd.get("limitation"), 200),
            "topics": _names(sd.get("topics")) or _names(sd.get("themes")),
            "year_start": ys, "year_end": ye, "countries": countries, "year_label": "years"}


def _document(m: dict) -> dict:
    d = m.get("document_description") or {}
    ts = d.get("title_statement") or {}
    countries, other = split_places(d.get("ref_country"))
    date = str(d.get("date_published") or d.get("date_created") or d.get("date_available") or "")[:10]
    where = ", ".join(_names(d.get("ref_country"))) or d.get("spatial_coverage") or None
    y = _year(date)
    return {"kind": "document", "id": ts.get("idno"), "name": ts.get("title") or "",
            "doc_type": d.get("type") or None, "date": date, "country": _clip(where, 80) if where else None,
            "abstract": _clip(d.get("abstract") or d.get("description"), 550), "passage": None,
            "countries": countries or other, "year_start": y, "year_end": y, "frequency": None,
            "year_label": "publication year", "_fulltext": None}


# ---------- types the model has not been trained on ----------
def _table(m: dict) -> dict:
    d = m.get("table_description") or {}
    ts = d.get("title_statement") or {}
    countries, other = split_places((d.get("ref_country") or []) + (d.get("geographic_units") or []))
    ys, ye = _span(d.get("time_periods"), "from", "to")
    if ys is None:
        ys, ye = _span(d.get("time_periods"))
    title = ts.get("title") or ""
    if ts.get("sub_title"):
        title += f" {ts['sub_title']}"
    return {"kind": "table", "id": ts.get("idno"), "name": title, "table_number": ts.get("table_number"),
            "producer": _join(d.get("authoring_entity") or d.get("publisher"), 120) or None,
            "date": str(d.get("date_published") or d.get("date_created") or "")[:10] or None,
            "columns": _join(d.get("table_columns"), 300, "label") or None, "rows": _join(d.get("table_rows"), 200, "label") or None,
            "unit_of_analysis": _join(d.get("unit_observation"), 120) or None, "universe": _join(d.get("universe"), 160, "value") or None,
            "granularity": d.get("geographic_granularity") or None, "geography": _clip(", ".join(other), 120) or None,
            "sources": _join(d.get("data_sources"), 200, "source") or None,
            "topics": (_names(d.get("topics")) or _names(d.get("themes")))[:6], "tags": _names(d.get("keywords"))[:8],
            "abstract": _clip(d.get("description") or " ".join(_names(d.get("table_footnotes"), "text")), 450),
            "countries": countries, "year_start": ys, "year_end": ye, "year_label": "years"}


def _geospatial(m: dict) -> dict:
    d = m.get("description") or {}
    ii = d.get("identificationInfo") or {}
    ii = ii[0] if isinstance(ii, list) and ii else ii
    cit = ii.get("citation") or {}
    kws = [k for k in ii.get("descriptiveKeywords") or [] if isinstance(k, dict)]
    place_kw = [k.get("keyword") for k in kws if (k.get("type") or "").lower() == "place" and k.get("keyword")]
    theme_kw = [k.get("keyword") for k in kws if (k.get("type") or "").lower() != "place" and k.get("keyword")]
    ext = ii.get("extent") or {}
    ext = ext[0] if isinstance(ext, list) and ext else ext
    geo_desc = [g.get("geographicDescription") for g in ext.get("geographicElement") or [] if isinstance(g, dict) and g.get("geographicDescription")]
    countries, other = split_places(geo_desc + place_kw)
    if not countries:
        countries = countries_in_text(cit.get("title") or "", *place_kw, *geo_desc)
    ys, ye = _span(ext.get("temporalElementExtent") or [], "beginPosition", "endPosition")
    if ys is None:
        dates = {x.get("type"): x.get("date") for x in cit.get("date") or [] if isinstance(x, dict)}
        y = _year(dates.get("creation") or dates.get("publication") or dates.get("released") or next(iter(dates.values()), None))
        ys = ye = y
    maint = (ii.get("resourceMaintenance") or [{}])[0]
    freq = maint.get("maintenanceAndUpdateFrequency") or maint.get("maintenanceOrUpdateFrequency")
    fmts = _names(ii.get("resourceFormat") or ii.get("resourceFormats"))
    org = next(iter(_names(cit.get("citedResponsibleParty"), "organisationName")), None) or ii.get("credit")
    return {"kind": "geospatial", "id": d.get("idno"), "name": cit.get("title") or "",
            "producer": _clip(org, 120) or None, "representation": _clip(", ".join(_names(ii.get("spatialRepresentationType"))), 60) or None,
            "formats": fmts[:5], "frequency": _clip(re.sub(r"\s*\[.*?\]", "", freq or ""), 40) or None,
            "geography": _clip(", ".join(other or geo_desc), 120) or None,
            "topics": [re.sub(r"\s*\[.*?\]", "", t) for t in _names(ii.get("topicCategory"))][:4], "tags": theme_kw[:8],
            "abstract": _clip(ii.get("abstract"), 450), "purpose": _clip(ii.get("purpose"), 200) or None,
            "countries": countries, "year_start": ys, "year_end": ye, "year_label": "years"}


def _image(m: dict) -> dict:
    d = m.get("image_description") or {}
    ip = (d.get("iptc") or {}).get("photoVideoMetadataIPTC") or {}
    dc = d.get("dcmi") or {}
    locs = ip.get("locationsShown") or ip.get("locationCreated") or []
    locs = locs if isinstance(locs, list) else [locs]
    cn = [{"name": l.get("countryName"), "code": l.get("countryCode")} for l in locs if isinstance(l, dict)]
    countries, _ = split_places(cn + (dc.get("country") or []))
    sub = [x for l in locs if isinstance(l, dict) for x in (l.get("city"), l.get("provinceState"), l.get("sublocation")) if x]
    date = str(ip.get("dateCreated") or dc.get("date") or "")[:10]
    y = _year(date)
    kws = [k for s in _names(ip.get("keywords")) + _names(dc.get("keywords")) for k in re.split(r"\s*,\s*", s) if k]
    return {"kind": "image", "id": d.get("idno"), "name": ip.get("headline") or ip.get("title") or dc.get("title") or "",
            "creator": _join(ip.get("creatorNames") or ([dc["creator"]] if dc.get("creator") else []), 100) or None,
            "producer": ip.get("source") or ip.get("creditLine") or dc.get("publisher") or None,
            "date": date or None, "place": _clip(", ".join(dict.fromkeys(sub)) or dc.get("coverage"), 100) or None,
            "album": _join(d.get("album"), 100) or None, "tags": list(dict.fromkeys(kws))[:8],
            "topics": _names(dc.get("topics"))[:4],
            "abstract": _clip(ip.get("description") or dc.get("description") or dc.get("caption"), 400),
            "countries": countries, "year_start": y, "year_end": y, "year_label": "date taken"}


def _video(m: dict) -> dict:
    d = m.get("video_description") or {}
    countries, other = split_places(d.get("country"))
    date = str(d.get("date_published") or d.get("date_created") or "")[:10]
    y = _year(date)
    dur = d.get("duration") or ""
    mm = re.match(r"PT(?:(\d+)H)?(?:(\d+)M)?(?:(\d+)S)?", dur)
    if mm and any(mm.groups()):
        h, mi, s = (int(x or 0) for x in mm.groups())
        dur = f"{h}h {mi}m" if h else f"{mi} min" if mi else f"{s} s"
    transcript = " ".join(t.get("text") or "" for t in d.get("transcript") or [] if isinstance(t, dict))
    transcript = re.sub(r"(?m)^\d+:\d+(:\d+)?\s*$", " ", transcript)
    return {"kind": "video", "id": d.get("idno"), "name": d.get("title") or "",
            "producer": d.get("creator") or d.get("publisher") or d.get("production_company") or None,
            "genre": d.get("genre") or None, "date": date or None, "duration": dur or None,
            "languages": _join(d.get("languages") or d.get("language"), 60) or None,
            "geography": _clip(d.get("spatial_coverage") or ", ".join(other), 100) or None,
            "reference_time": d.get("content_reference_time") or d.get("temporal_coverage") or None,
            "tags": _names(d.get("keywords"))[:8], "topics": _names(d.get("topics"))[:4],
            "abstract": _clip(d.get("description"), 450), "passage": None, "_fulltext": _clip(transcript, 400_000) or None,
            "countries": countries, "year_start": y, "year_end": y, "year_label": "publication year"}


def _script(m: dict) -> dict:
    d = m.get("project_desc") or {}
    ts = d.get("title_statement") or {}
    countries, other = split_places(d.get("geographic_units"))
    y = _year(d.get("production_date") or (m.get("doc_desc") or {}).get("prod_date"))
    return {"kind": "script", "id": ts.get("idno"), "name": ts.get("title") or "",
            "producer": _join(d.get("authoring_entity"), 140) or None,
            "datasets": _join(d.get("datasets"), 240) or None, "software": _join(d.get("software"), 100) or None,
            "methods": _join(d.get("methods"), 160) or None, "geography": _clip(", ".join(other), 100) or None,
            "topics": (_names(d.get("topics")) or _names(d.get("themes")))[:6], "tags": _names(d.get("keywords"))[:8],
            "abstract": _clip(d.get("abstract"), 500), "countries": countries,
            "year_start": y, "year_end": y, "year_label": "production year"}


def _database(m: dict) -> dict:
    d = m.get("database_description") or {}
    ts = d.get("title_statement") or {}
    countries, other = split_places((d.get("ref_country") or []) + (d.get("geographic_units") or []))
    ys, ye = _span(d.get("time_coverage"))
    alt = ts.get("alternate_title")
    return {"kind": "database", "id": ts.get("idno"), "name": f"{ts.get('title') or ''}{' (' + alt + ')' if alt else ''}",
            "producer": _join(d.get("authoring_entity"), 140) or None,
            "frequency": _join(d.get("periodicity"), 60, "period") or None, "update_frequency": d.get("update_frequency") or None,
            "indicators": len(d.get("indicators") or []) or None, "granularity": d.get("geographic_granularity") or None,
            "geography": _clip(d.get("geographic_coverage_note") or "", 120) or None,
            "topics": (_names(d.get("topics")) or _names(d.get("themes")))[:12],
            "abstract": _clip(d.get("abstract"), 500), "countries": countries,
            "year_start": ys, "year_end": ye, "year_label": "years"}


_BUILDERS = {"microdata": _microdata, "timeseries": _timeseries, "document": _document, "table": _table,
             "geospatial": _geospatial, "image": _image, "video": _video, "script": _script, "timeseriesdb": _database}


# ---------- renderers for the new kinds (same line style as synth.sources) ----------
# The trained renderers print every field and drop a line whose last field is empty; the new ones print only the
# fields that are present, so a missing value never hides the others on its line.
def _years(rec: dict) -> str | None:
    ys, ye = rec.get("year_start"), rec.get("year_end")
    return f"{ys}-{ye}" if ys and ye and ys != ye else str(ys) if ys else None


def _cov(rec: dict) -> str | None:
    if rec.get("countries_truncated"):
        return "multi-country, full list not shown"
    return _coverage(rec["countries"]) if rec.get("countries") else None


def _line(*pairs) -> str | None:
    parts = [f"{k}: {v}" for k, v in pairs if v not in (None, "", [])]
    return "; ".join(parts) or None


def _lines(*ls) -> list[str]:
    return [l for l in ls if l]


def _kw(rec: dict) -> str | None:
    return ", ".join(dict.fromkeys(rec.get("topics", []) + rec.get("tags", []))) or None


class Table:
    kind, label = "table", "statistical table"

    def render(self, rec):
        num = f" (table {rec['table_number']})" if rec.get("table_number") else ""
        return _lines(f"Result: {rec['name']}{num}",
                      _line(("Producer", rec.get("producer")), ("Published", rec.get("date"))),
                      _line(("Years", _years(rec)), ("Geographic granularity", rec.get("granularity"))),
                      _line(("Coverage", _cov(rec)), ("Geographic scope", rec.get("geography"))),
                      _line(("Columns", rec.get("columns"))), _line(("Rows", rec.get("rows"))),
                      _line(("Unit of observation", rec.get("unit_of_analysis")), ("Universe", rec.get("universe"))),
                      _line(("Topics", _kw(rec))), _line(("Data sources", rec.get("sources"))),
                      _line(("Description", rec.get("abstract"))))


class Geospatial:
    kind, label = "geospatial", "geospatial dataset (map layer)"

    def render(self, rec):
        return _lines(f"Result: {rec['name']}",
                      _line(("Producer", rec.get("producer")), ("Representation", rec.get("representation")),
                            ("Formats", ", ".join(rec.get("formats") or []))),
                      _line(("Years", _years(rec)), ("Update frequency", rec.get("frequency"))),
                      _line(("Coverage", _cov(rec)), ("Geographic scope", rec.get("geography"))),
                      _line(("Topics", _kw(rec))), _line(("Description", rec.get("abstract"))),
                      _line(("Purpose", rec.get("purpose"))))


class Image:
    kind, label = "image", "photograph"

    def render(self, rec):
        return _lines(f"Result: {rec['name']}",
                      _line(("Photographer", rec.get("creator")), ("Source", rec.get("producer")), ("Album", rec.get("album"))),
                      _line(("Date taken", rec.get("date"))),
                      _line(("Coverage", _cov(rec)), ("Place shown", rec.get("place"))),
                      _line(("Keywords", _kw(rec))), _line(("Caption", rec.get("abstract"))))


class Video:
    kind, label = "video", "video"

    def render(self, rec):
        return _lines(f"Result: {rec['name']}",
                      _line(("Producer", rec.get("producer")), ("Genre", rec.get("genre")), ("Duration", rec.get("duration")),
                            ("Language", rec.get("languages"))),
                      _line(("Published", rec.get("date")), ("Period shown", rec.get("reference_time"))),
                      _line(("Coverage", _cov(rec)), ("Geographic scope", rec.get("geography"))),
                      _line(("Keywords", _kw(rec))), _line(("Description", rec.get("abstract"))),
                      _line(("Transcript excerpt", rec.get("passage"))))


class Script:
    kind, label = "script", "research project with code and data (reproducibility package)"

    def render(self, rec):
        return _lines(f"Result: {rec['name']}",
                      _line(("Authors", rec.get("producer")), ("Year", _years(rec))),
                      _line(("Coverage", _cov(rec)), ("Geographic scope", rec.get("geography"))),
                      _line(("Data used", rec.get("datasets"))),
                      _line(("Methods", rec.get("methods")), ("Software", rec.get("software"))),
                      _line(("Topics", _kw(rec))), _line(("Abstract", rec.get("abstract"))))


class Database:
    kind, label = "database", "time series database"

    def render(self, rec):
        return _lines(f"Result: {rec['name']}",
                      _line(("Producer", rec.get("producer")), ("Indicators", rec.get("indicators"))),
                      _line(("Years", _years(rec)), ("Periodicity", rec.get("frequency")), ("Updated", rec.get("update_frequency"))),
                      _line(("Coverage", _cov(rec)), ("Geographic granularity", rec.get("granularity"))),
                      _line(("Topics", _kw(rec))), _line(("Abstract", rec.get("abstract"))))


RENDERERS = {**SOURCES, **{c.kind: c() for c in (Table, Geospatial, Image, Video, Script, Database)}}
TRAINED_KINDS = set(SOURCES)

_U = None
_ST = None


def similarity(search: str, rec: dict) -> float:
    """Cosine similarity with the retriever used in training (CPU). Pass your own score to to_input to skip this."""
    global _ST
    from .synth.signals import MODEL, item_text
    if _ST is None:
        from sentence_transformers import SentenceTransformer
        _ST = SentenceTransformer(MODEL, device="cpu")
    a, b = _ST.encode([search, item_text(rec)], normalize_embeddings=True)
    return float(a @ b)


def to_input(rec: dict, search: str, language: str = "English", facets: dict | None = None,
             sim: float | None = None, fmt: str = "itemfirst") -> str:
    """The model input for a record (from to_record) and a search. fmt: 'itemfirst' (current models) or 'full'."""
    global _U
    from .synth.realistic import _clean_item_lines, item_first
    from .synth.understand import Understander, check
    if _U is None:
        _U = Understander()
    rec = dict(rec)
    if rec.get("_fulltext") and not rec.get("passage"):
        rec["passage"] = Documents.passage(rec["_fulltext"], search)
    src = RENDERERS[rec["kind"]]
    facets = facets or {}
    parsed = _U.parse(search, facets)
    sim = similarity(search, rec) if sim is None else sim
    fl = "; ".join(f"{k}={'/'.join(v) if isinstance(v, list) else v}" for k, v in facets.items()) or "none"
    chk = check(rec, parsed, _U)
    lines = [f"Respond in: {language}", f"Item type: {src.label}", f"Search: {search}", f"Facets: {fl}",
             f"Understood: {_U.describe(parsed)}", "Checks: " + (" | ".join(chk) if chk else "none"),
             f"Retrieval similarity: {sim:.2f}", *_clean_item_lines(src.render(rec))]
    v2 = "\n".join(lines)
    return item_first(v2) if fmt == "itemfirst" else v2
