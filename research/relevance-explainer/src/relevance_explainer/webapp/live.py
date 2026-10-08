"""Live catalog records for the web app: fetch one record from a Microdata Library (or other NADA) URL or a Data360
indicator URL, convert it with the metadata-schemas adapter and render the model input for a typed search.

Only the hosts in NADA_HOSTS and the Data360 API are contacted, so the endpoint cannot be used to fetch arbitrary URLs.
"""
from __future__ import annotations

import json
import re
import time
import urllib.parse
import urllib.request
from functools import lru_cache

NADA_HOSTS = {"microdata.worldbank.org", "catalog.ihsn.org", "nada-demo.ihsn.org", "reproducibility.worldbank.org"}
D360_SEARCH = "https://data360api.worldbank.org/data360/searchv2"
UA = {"User-Agent": "relevance-explainer demo", "Content-Type": "application/json"}
LANGS = ["English", "French", "Spanish", "Portuguese", "Arabic", "Chinese", "Russian", "Indonesian", "Hindi", "Turkish", "Swahili"]


class LiveError(ValueError):
    pass


def parse_url(url: str) -> tuple[str, str, str]:
    """Return (source, host, identifier) for a supported catalog link, or a bare Data360 indicator id."""
    url = url.strip()
    if re.fullmatch(r"[A-Z][A-Z0-9]*_[A-Z0-9_.]+", url):
        return "data360", "data360", url
    u = urllib.parse.urlparse(url if "://" in url else "https://" + url)
    host = (u.hostname or "").lower()
    if host in NADA_HOSTS:
        m = re.search(r"/catalog/(\d+)(?:/|$)", u.path)
        if m:
            return "nada-id", host, m.group(1)
        m = re.search(r"/catalog/([A-Za-z0-9_.\-]+)(?:/|$)", u.path)
        if m and m.group(1) not in ("search", "export"):
            return "nada-idno", host, m.group(1)
        raise LiveError("This looks like a catalog page, but not a single study. Open the study and copy its link.")
    if host.endswith("data360.worldbank.org"):
        m = re.search(r"([A-Z][A-Z0-9]*_[A-Z0-9_.]+)", urllib.parse.unquote(u.path + "?" + u.query))
        if m:
            return "data360", host, m.group(1)
        raise LiveError("No Data360 indicator id found in this link (for example WB_WDI_SP_POP_TOTL).")
    raise LiveError(f"Supported links: Data360 indicators and studies in {', '.join(sorted(NADA_HOSTS))}.")


def _get(url: str, post: dict | None = None) -> dict:
    data = json.dumps(post).encode() if post is not None else None
    req = urllib.request.Request(url, data, UA if data else {"User-Agent": UA["User-Agent"]})
    with urllib.request.urlopen(req, timeout=25) as r:
        return json.loads(r.read())


@lru_cache(maxsize=256)
def fetch(url: str) -> tuple[dict, str]:
    """(record, canonical page link) for a supported link. Cached per link."""
    from ..schemas import to_record
    src, host, ident = parse_url(url)
    try:
        if src.startswith("nada"):
            q = "?id_format=id" if src == "nada-id" else ""
            d = _get(f"https://{host}/index.php/api/catalog/{urllib.parse.quote(ident)}{q}")
            if d.get("status") != "success" or "dataset" not in d:
                raise LiveError("The catalog did not return this study.")
            rec = to_record(d)
            page = f"https://{host}/index.php/catalog/{d['dataset'].get('id', ident)}"
        else:
            res = _get(D360_SEARCH, {"search": "*", "top": 1, "count": True, "filter": f"series_description/idno eq '{ident}'"})
            hits = res.get("value") or []
            if not hits:
                raise LiveError(f"Data360 has no indicator {ident}.")
            rec = to_record(hits[0])
            page = f"https://data360.worldbank.org/en/indicator/{ident}"
    except LiveError:
        raise
    except Exception as e:
        raise LiveError(f"Could not fetch the record: {type(e).__name__}") from e
    if not rec.get("name"):
        raise LiveError("The record has no title; it may not be public.")
    return rec, page


def detect_language(text: str) -> str:
    """The search's language among the eleven the model writes: script first, then fastText; English otherwise."""
    from ..synth import realistic as R
    for lang, pat in R.SCRIPT.items():
        if re.search(pat, text):
            return lang
    try:
        if R._LID is None and R.LID_PATH.exists():
            import fasttext
            R._LID = fasttext.load_model(str(R.LID_PATH))
        pred = R._LID.f.predict(text.replace("\n", " "), 1, 0.0, "strict") if R._LID is not None else []
    except Exception:
        pred = []
    if pred:
        p, label = pred[0]
        code = label.replace("__label__", "")
        from ..synth.geo import LANG_CODES
        for lang, c in LANG_CODES.items():
            if c == code and p >= 0.4:
                return lang
    return "English"


def render(url: str, search: str, language: str | None) -> dict:
    from ..schemas import RENDERERS, similarity, to_input
    rec, page = fetch(url)
    lang = language if language in LANGS else detect_language(search)
    t0 = time.time()
    sim = similarity(search, rec)
    text = to_input(rec, search, lang, sim=sim, fmt="itemfirst", render_version=3)
    return {"input": text, "language": lang, "detected": language not in LANGS, "similarity": round(sim, 2),
            "record": {"title": rec["name"], "type": RENDERERS[rec["kind"]].label, "kind": rec["kind"], "link": page,
                       "countries": len(rec.get("countries") or []), "years": [rec.get("year_start"), rec.get("year_end")]},
            "render_ms": int((time.time() - t0) * 1000)}
