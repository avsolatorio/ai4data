"""Open-access catalog sources. Each source exposes search -> hydrate -> render.

Record kinds: indicator (Data360), study (Microdata Library), document (World Bank
Documents & Reports, with a passage cut from the full text), dataset (HDX).
Every hydrated record carries the common keys used by `filter_check`:
countries, year_start, year_end, frequency, year_label.
"""
from __future__ import annotations

import json
import re
import unicodedata
import urllib.parse
import urllib.request

from . import data360

UA = {"User-Agent": "curl/8.5.0"}
STOP = set("the and for with from that this are was were has have not but all any can its into than then their about "
           "which what when where how who why per rate level data".split())


def _get(url: str, *, post: dict | None = None, timeout: int = 60, limit: int | None = None) -> bytes:
    headers = {**UA, **({"Content-Type": "application/json"} if post else {})}
    req = urllib.request.Request(url, json.dumps(post).encode() if post else None, headers)
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return r.read(limit) if limit else r.read()


def _clip(s: str | None, n: int) -> str:
    s = re.sub(r"\s+", " ", s or "").strip()
    return s if len(s) <= n else s[: n - 1].rstrip() + "…"


def _year(s: str | None) -> int | None:
    m = re.search(r"(19|20)\d{2}", s or "")
    return int(m.group(0)) if m else None


# ---------- Data360 indicators ----------
class Data360:
    kind = "indicator"
    label = "statistical indicator series"

    def search(self, q: str, top: int) -> list[dict]:
        return data360.search(q, top=top)

    def hydrate(self, hit: dict, q: str) -> dict | None:
        rec = data360.compact(hit)
        return {**rec, "kind": self.kind, "year_label": "years"} if rec else None

    def render(self, rec: dict) -> list[str]:
        span = f"{rec['year_start']}-{rec['year_end']}" if rec["year_start"] else "not stated"
        lines = [
            f"Result: {rec['name']}",
            f"Database: {rec['database'] or 'n/a'}; Source: {rec['source'] or 'n/a'}",
            f"Unit: {rec['unit'] or 'n/a'}; Frequency: {rec['frequency'] or 'n/a'}; Years: {span}",
            f"Coverage: {_coverage(rec['countries'])}",
            f"Topics: {', '.join(rec['topics']) or 'n/a'}",
            f"Definition: {rec['definition'] or 'n/a'}",
        ]
        if rec["concept"]:
            lines.append(f"Concept: {rec['concept']}")
        if rec["limitation"]:
            lines.append(f"Limitations: {rec['limitation']}")
        return lines


# ---------- Microdata Library studies ----------
class Microdata:
    kind = "study"
    label = "survey microdata study"
    BASE = "https://microdata.worldbank.org/index.php/api/catalog"

    def search(self, q: str, top: int) -> list[dict]:
        url = f"{self.BASE}/search?format=json&sort_by=rank&ps={top}&sk={urllib.parse.quote(q)}"
        return json.loads(_get(url))["result"]["rows"]

    def hydrate(self, hit: dict, q: str) -> dict | None:
        d = json.loads(_get(f"{self.BASE}/{urllib.parse.quote(hit['idno'])}"))
        d = d.get("dataset", d)
        sd = (d.get("metadata") or {}).get("study_desc") or {}
        info, method = sd.get("study_info") or {}, sd.get("method") or {}
        title = (sd.get("title_statement") or {}).get("title") or hit["title"]
        sub = (sd.get("title_statement") or {}).get("sub_title")
        countries = sorted({n["name"] for n in info.get("nation") or [] if n.get("name")})
        # The search row lists more countries than the detail record for multi-country studies,
        # but truncates long lists ("..." or "and N more").
        row = hit.get("nation") or ""
        truncated = "..." in row
        listed = [c.strip() for c in re.sub(r"\.\.\.and \d+ more", "", row).split(",")]
        listed = [c for c in listed if c and "..." not in c]  # also drops a name cut off mid-word
        if len(set(listed)) > len(countries):
            countries = sorted(set(listed))
        else:
            truncated = False
        coll = (info.get("coll_dates") or [{}])[0]
        mode = (method.get("data_collection") or {}).get("coll_mode") or []
        return {
            "kind": self.kind, "id": hit["idno"], "name": f"{title}{' (' + sub + ')' if sub else ''}",
            "producer": _clip(hit.get("authoring_entity") or hit.get("repo_title"), 100),
            "data_kind": _clip(info.get("data_kind"), 60), "unit_of_analysis": _clip(info.get("analysis_unit"), 80),
            "universe": _clip(info.get("universe"), 200), "geography": _clip(info.get("geog_coverage"), 120),
            "collection_mode": _clip(", ".join(mode) if isinstance(mode, list) else mode, 80),
            "topics": [t["topic"] for t in info.get("topics") or [] if t.get("topic")][:6],
            "abstract": _clip(info.get("abstract"), 500), "varcount": hit.get("varcount"),
            "countries": countries, "countries_truncated": truncated,
            "year_start": hit.get("year_start") or _year(coll.get("start")),
            "year_end": hit.get("year_end") or _year(coll.get("end")),
            "frequency": None, "year_label": "survey years",
        }

    def render(self, rec: dict) -> list[str]:
        span = f"{rec['year_start']}-{rec['year_end']}" if rec["year_start"] else "not stated"
        return [
            f"Result: {rec['name']}",
            f"Producer: {rec['producer'] or 'n/a'}; Type: {rec['data_kind'] or 'n/a'}; Variables: {rec['varcount'] if rec['varcount'] is not None else 'n/a'}",
            f"Survey years: {span}; Collection mode: {rec['collection_mode'] or 'n/a'}",
            f"Coverage: {'multi-country, full list not shown' if rec['countries_truncated'] else _coverage(rec['countries'])}; Geographic scope: {rec['geography'] or 'n/a'}",
            f"Unit of analysis: {rec['unit_of_analysis'] or 'n/a'}; Universe: {rec['universe'] or 'n/a'}",
            f"Topics: {', '.join(rec['topics']) or 'n/a'}",
            f"Abstract: {rec['abstract'] or 'n/a'}",
        ]


# ---------- World Bank Documents & Reports (Policy Research Working Papers) ----------
class Documents:
    kind = "document"
    label = "research document (Policy Research Working Papers and related reports)"
    URL = "https://search.worldbank.org/api/v3/wds"
    FL = "id,docdt,display_title,abstracts,docty,count,txturl,lang_exact"

    def search(self, q: str, top: int) -> list[dict]:
        qs = urllib.parse.urlencode({"format": "json", "qterm": q, "rows": top, "fl": self.FL,
                                     "lang_exact": "English", "docty_exact": "Policy Research Working Paper"})
        docs = json.loads(_get(f"{self.URL}?{qs}")).get("documents") or {}
        return [d for d in docs.values() if isinstance(d, dict) and d.get("txturl")]

    @staticmethod
    def passage(text: str, query: str) -> str | None:
        words = text.split()
        terms = {w for w in re.findall(r"[a-z]{4,}", query.lower()) if w not in STOP}
        if not terms:
            return None
        best, best_score = None, 0
        # Skip the cover page and abstract, which repeat the metadata abstract.
        first = 700 if len(words) > 4000 else 0
        for i in range(first, max(len(words) - 110, first + 1), 55):
            chunk = " ".join(words[i : i + 110])
            letters = sum(c.isalpha() for c in chunk)
            if letters / max(len(chunk), 1) < 0.78 or re.search(r"references|bibliography|table of contents", chunk, re.I):
                continue
            low = chunk.lower()
            score = sum(1 for t in terms if t in low)
            if score > best_score:
                best, best_score = chunk, score
        return _clip(best, 700) if best and best_score >= max(2, len(terms) // 3) else None

    def hydrate(self, hit: dict, q: str) -> dict | None:
        abstract = (hit.get("abstracts") or {}).get("cdata!") or ""
        try:
            text = _get(hit["txturl"], timeout=60, limit=400_000).decode("utf-8", "replace")
        except Exception:
            text = ""
        year = _year(hit.get("docdt"))
        return {
            "kind": self.kind, "id": str(hit["id"]), "name": hit["display_title"],
            "doc_type": hit.get("docty"), "date": (hit.get("docdt") or "")[:10],
            "country": hit.get("count"), "abstract": _clip(abstract, 550),
            "passage": self.passage(text, q) if text else None,
            "countries": [hit["count"]] if hit.get("count") else [],
            "year_start": year, "year_end": year, "frequency": None, "year_label": "publication year",
        }

    def render(self, rec: dict) -> list[str]:
        return [
            f"Result: {rec['name']}",
            f"Document type: {rec['doc_type'] or 'n/a'}; Published: {rec['date'] or 'n/a'}; Country or region: {rec['country'] or 'n/a'}",
            f"Abstract: {rec['abstract'] or 'n/a'}",
            f"Retrieved passage: {rec['passage'] or 'none retrieved'}",
        ]


# ---------- Humanitarian Data Exchange (CKAN) ----------
class HDX:
    kind = "dataset"
    label = "humanitarian dataset (HDX)"
    URL = "https://data.humdata.org/api/3/action/package_search"

    def search(self, q: str, top: int) -> list[dict]:
        qs = urllib.parse.urlencode({"q": q, "rows": top})
        return json.loads(_get(f"{self.URL}?{qs}"))["result"]["results"]

    def hydrate(self, hit: dict, q: str) -> dict | None:
        if not hit.get("title"):
            return None
        dates = re.findall(r"(\d{4})-\d{2}-\d{2}", hit.get("dataset_date") or "")
        days = hit.get("data_update_frequency")
        freq = {"-1": "never updated", "-2": "as needed", "0": "live", "1": "daily", "7": "weekly", "14": "every two weeks",
                "30": "monthly", "90": "quarterly", "180": "every six months", "365": "annual"}.get(str(days))
        return {
            "kind": self.kind, "id": hit.get("name") or hit["id"], "name": hit["title"],
            "organization": (hit.get("organization") or {}).get("title"), "source": hit.get("dataset_source"),
            "notes": _clip(hit.get("notes"), 500), "tags": [t["name"] for t in hit.get("tags") or []][:8],
            "formats": sorted({r.get("format") for r in hit.get("resources") or [] if r.get("format")})[:5],
            "countries": sorted(g["title"] for g in hit.get("groups") or [] if g.get("title")),
            "year_start": int(dates[0]) if dates else None, "year_end": int(dates[1]) if len(dates) > 1 else None,
            "frequency": freq, "update_frequency": freq, "year_label": "years",
        }

    def render(self, rec: dict) -> list[str]:
        span = f"{rec['year_start']}-{rec['year_end']}" if rec["year_start"] and rec["year_end"] else "not stated"
        return [
            f"Result: {rec['name']}",
            f"Organization: {rec['organization'] or 'n/a'}; Source: {rec['source'] or 'n/a'}; Formats: {', '.join(rec['formats']) or 'n/a'}",
            f"Years: {span}; Update frequency: {rec['update_frequency'] or 'n/a'}",
            f"Coverage: {_coverage(rec['countries'])}",
            f"Tags: {', '.join(rec['tags']) or 'n/a'}",
            f"Description: {rec['notes'] or 'n/a'}",
        ]


SOURCES = {s.kind: s for s in (Data360(), Microdata(), Documents(), HDX())}


# ---------- local record pools (synth/pools.py) for the metadata-schemas types ----------
class PoolSource:
    """Search over a local pool of records with the training retriever. Enabled only through RECORD_POOLS, a JSON
    object {kind: [pool.jsonl, ...]}, so the catalog sources above are unchanged when it is not set."""

    def __init__(self, kind: str, paths: list[str]):
        self.kind = kind
        self.recs = [json.loads(l) for p in paths for l in open(p, encoding="utf-8") if l.strip()]
        self._emb = None
        # One lock for every pool: they share one embedding model, and concurrent encodes on it failed with dtype errors
        # (2026-10-07; most of the anchored and retrieval stages of type chunks c3, c4, c5 and c8 were lost).
        self._lock = _POOL_LOCK

    @property
    def label(self) -> str:
        from ..schemas import RENDERERS
        return RENDERERS[self.kind].label

    def render(self, rec: dict) -> list[str]:
        from ..schemas import RENDERERS
        return RENDERERS[self.kind].render(rec)

    def _model(self):
        global _POOL_MODEL
        if _POOL_MODEL is None:
            from sentence_transformers import SentenceTransformer
            from .signals import MODEL
            # CPU: the pool search runs next to vLLM and training on the GPUs, where it hit allocation failures
            _POOL_MODEL = SentenceTransformer(MODEL, device=__import__("os").environ.get("POOL_DEVICE", "cpu"))
        return _POOL_MODEL

    def search(self, q: str, top: int) -> list[dict]:
        import numpy as np
        from .signals import item_text
        with self._lock:
            if self._emb is None:
                self._emb = self._model().encode([item_text(r) for r in self.recs], batch_size=256, normalize_embeddings=True,
                                                 show_progress_bar=False)
            qv = self._model().encode([q], normalize_embeddings=True, show_progress_bar=False)[0]
        s = self._emb @ qv
        return [{"_i": int(i)} for i in np.argsort(-s)[:top]]

    def hydrate(self, hit: dict, q: str) -> dict | None:
        return dict(self.recs[hit["_i"]])


_POOL_MODEL = None
_POOL_LOCK = __import__("threading").Lock()
if __import__("os").environ.get("RECORD_POOLS"):
    for _kind, _paths in json.loads(__import__("os").environ["RECORD_POOLS"]).items():
        SOURCES[_kind] = PoolSource(_kind, _paths)


def _coverage(countries: list[str]) -> str:
    n = len(countries)
    if n == 0:
        return "not stated"
    if n <= 12:
        return f"{n} {'country' if n == 1 else 'countries'}: " + ", ".join(countries)
    return f"{n} countries and economies"


_ALIASES = {
    "turkiye": "turkey", "viet nam": "vietnam", "russian federation": "russia", "lao pdr": "laos",
    "united republic of tanzania": "tanzania", "korea, rep.": "south korea", "republic of korea": "south korea",
    "korea, dem. people's rep.": "north korea", "iran, islamic rep.": "iran", "egypt, arab rep.": "egypt",
    "egypt, arab republic of": "egypt", "yemen, rep.": "yemen", "venezuela, rb": "venezuela",
    "congo, dem. rep.": "democratic republic of the congo", "congo, rep.": "republic of the congo",
    "gambia, the": "gambia", "bahamas, the": "bahamas", "kyrgyz republic": "kyrgyzstan", "slovak republic": "slovakia",
    "syrian arab republic": "syria", "cote d'ivoire": "ivory coast", "czechia": "czech republic",
    "west bank and gaza": "palestine", "state of palestine": "palestine", "myanmar (burma)": "myanmar",
}
_REGIONS = {"world", "africa", "east asia and pacific", "south asia", "latin america and caribbean",
            "europe and central asia", "middle east and north africa", "sub-saharan africa", "eastern and southern africa",
            "western and central africa", "europe", "asia", "middle east"}


def _norm_country(s: str) -> str:
    s = unicodedata.normalize("NFKD", s).encode("ascii", "ignore").decode().lower().strip()
    s = re.sub(r"\s*\(.*?\)", "", s)
    return _ALIASES.get(s, s)


def _norm_text(s: str) -> str:
    s = unicodedata.normalize("NFKD", s).encode("ascii", "ignore").decode().lower()
    return re.sub(r"\s+", " ", s)


def _same_country(a: str, b: str) -> bool:
    a, b = _norm_country(a), _norm_country(b)
    return a == b or a.split(",")[0] == b.split(",")[0]


def filter_check(rec: dict, filters: dict) -> list[str]:
    """Deterministic coverage checks for filters that can be verified from metadata."""
    out = []
    text = _norm_text(" ".join(str(rec.get(k) or "") for k in ("name", "abstract", "notes", "definition", "passage", "universe")))
    for c in filters.get("countries") or []:
        raw = _norm_text(" ".join(rec["countries"]))
        if not rec["countries"]:
            out.append(f"country {c}: coverage not stated")
        elif any(_same_country(c, x) for x in rec["countries"]) or (rec["kind"] == "document" and _norm_country(c) in raw):
            out.append(f"country {c}: covered")
        elif _norm_country(c) in text:
            out.append(f"country {c}: not in the coverage list but named in the description")
        elif rec.get("countries_truncated"):
            out.append(f"country {c}: cannot confirm (multi-country study, full list not shown)")
        elif rec["kind"] == "document" and _norm_country(rec["countries"][0]) in _REGIONS:
            out.append(f"country {c}: document covers a region or the world ({rec['countries'][0]})")
        else:
            out.append(f"country {c}: not covered")
    yrs, label = filters.get("years"), rec.get("year_label", "years")
    if yrs and rec["kind"] == "document":
        # A paper's publication year says nothing about the period its data covers.
        out.append(f"years {yrs}: period studied not verifiable from metadata (published {rec['year_start'] or 'date not stated'})")
    elif yrs:
        m = re.fullmatch(r"\s*(\d{4})\s*(?:-|–|to)?\s*(\d{4})?\s*", str(yrs))
        if m and rec["year_start"] and rec["year_end"]:
            a, b = int(m.group(1)), int(m.group(2) or m.group(1))
            point = rec["year_start"] == rec["year_end"]
            if rec["year_start"] <= a and b <= rec["year_end"]:
                status = "fully covered"
            elif b < rec["year_start"] or a > rec["year_end"]:
                status = "outside the range" if point else "not covered"
            elif point:
                status = "within the range"
            else:
                status = "partly covered"
            out.append(f"{label} {yrs}: {status}")
        else:
            out.append(f"{label} {yrs}: coverage not stated")
    freq = filters.get("frequency")
    if freq:
        if rec.get("frequency"):
            out.append(f"frequency {freq}: {'matches' if freq.lower() == rec['frequency'].lower() else 'differs (' + rec['frequency'] + ')'}")
        else:
            out.append(f"frequency {freq}: not stated")
    return out


def render_input(query: str, filters: dict, rec: dict, language: str = "English") -> str:
    """Text input shown to the teacher and, later, to the student model."""
    fl = "; ".join(f"{k}={'/'.join(v) if isinstance(v, list) else v}" for k, v in filters.items()) or "none"
    src = SOURCES[rec["kind"]]
    lines = [f"Respond in: {language}", f"Item type: {src.label}", f"Query: {query}", f"Filters: {fl}", *src.render(rec)]
    checks = filter_check(rec, filters)
    if checks:
        lines.append("Filter check: " + "; ".join(checks))
    return "\n".join(lines)
