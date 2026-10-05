"""Query understanding and constraint matching for realistic searches (no model weights).

`Understander.parse(query, facets)` finds places (countries, groupings, first-level subnational units, larger
cities), time (years, ranges, decades, relative expressions), frequency, population groups, requested data types and
generic scopes in the query text of any of the 11 languages, and merges interface facets. `check(rec, parsed)`
compares each constraint with a catalog record and returns one status line per constraint.

Resources: data/geo/geo.json (synth.geo) and data/geo/lexicon.json (synth.lexicon).
"""
from __future__ import annotations

import json
import re
import unicodedata
from functools import lru_cache
from pathlib import Path

_RES = Path(__file__).resolve().parents[3] / "resources"  # tracked copies; data/geo holds freshly built ones
GEO = Path("data/geo/geo.json") if Path("data/geo/geo.json").exists() else _RES / "geo.json"
LEX = Path("data/geo/lexicon.json") if Path("data/geo/lexicon.json").exists() else _RES / "lexicon.json"
COMMON = Path("data/geo/common_words.json")
ALLOW_PLACES = {"bihar", "gaza", "papua"}  # frequent in searches because people search for them
CURRENT_YEAR = 2026
YEAR = r"(?:19[5-9]\d|20[0-3]\d)"

# Place names that are also everyday words or too generic to mean a place in a search.
STOP_PLACES = {
    "male", "female", "of", "bar", "para", "mobile", "nice", "split", "bath", "reading", "batman", "mesa", "data", "central",
    "north", "south", "east", "west", "northern", "southern", "eastern", "western", "centre", "center", "capital", "coast",
    "national", "union", "federal", "district", "region", "province", "state", "city", "rural", "urban", "health", "rio",
    "san", "santa", "la", "el", "al", "as", "es", "da", "de", "do", "ba", "aba", "ede", "ife", "orlu", "gaza strip",
    "independencia", "libertad", "victoria", "concepcion", "trinidad", "esperanza", "progreso", "union", "colon", "bolivar",
    "sucre", "cuba", "chad", "jordan", "georgia", "niger", "guinea", "mali", "togo", "peru", "chile",
    "lima", "cuenca", "corona", "covid", "kota", "pueblo", "valle", "mina", "sur", "norte", "oeste", "este",
}
GROUP_EVIDENCE = {  # English metadata words that show a population group is covered
    "female": r"\b(women|woman|female|girls?|sex|gender|maternal|mothers?)\b",
    "male": r"\b(men|male|boys?|sex|gender)\b",
    "children": r"\b(child|children|under[- ]?5|under[- ]five|infants?|newborn|neonatal|pupils?|school[- ]age|stunting|wasting)\b",
    "youth": r"\b(youth|adolescen\w*|young (people|adults)|aged 15)\b",
    "elderly": r"\b(older|elderly|aged 6[05]|ageing|aging|pension\w*)\b",
    "urban": r"\b(urban|cities|city|towns?|metropolitan)\b",
    "rural": r"\b(rural|villages?|farm\w*|agricultur\w*)\b",
    "poorest": r"\b(poor|poorest|poverty|quintiles?|wealth|bottom 40|low[- ]income households?)\b",
    "displaced": r"\b(refugees?|displace\w*|idps?|asylum|migrants?|camps?)\b",
    "disability": r"\b(disabilit\w*|disabled|impairment)\b",
}
DATATYPE_KIND = {"indicator": "indicator", "microdata": "study", "document": "document", "dataset": "dataset"}
KIND_LABEL = {"indicator": "an indicator series", "study": "survey microdata", "document": "a document", "dataset": "a dataset",
              # metadata-schemas types without training data (relevance_explainer.schemas)
              "table": "a statistical table", "geospatial": "a geospatial dataset", "image": "a photograph",
              "video": "a video", "script": "a research project", "database": "a time series database"}


def norm(s: str) -> str:
    """Lowercase; drop accents on Latin letters only (Devanagari vowel signs and other combining marks are kept)."""
    out, prev_latin = [], False
    for ch in unicodedata.normalize("NFKD", s.lower().replace("ı", "i")):  # Turkish dotless i is often typed as i
        if unicodedata.combining(ch):
            if prev_latin:
                continue
        else:
            prev_latin = ch.isascii() or "LATIN" in unicodedata.name(ch, "")
        out.append(ch)
    return unicodedata.normalize("NFC", "".join(out))


def _variants(n: str) -> set[str]:
    """Query tokens plus forms without Arabic clitics (الريفية -> ريفية)."""
    out = set()
    for t in TOKEN.findall(n):
        out.add(t)
        for p in AR_PREFIX:
            if t.startswith(p) and len(t) - len(p) >= 3:
                out.add(t[len(p):])
    return out


def _has(n: str, toks: set[str], w: str) -> bool:
    """Lexicon entry w occurs in the normalized query n: exact word or phrase, Chinese substring, or an inflected form
    of a one-word entry (Latin plural or agreement endings, Russian case endings, Arabic clitics)."""
    if _is_cjk(w):
        return w in n
    if " " in w or "-" in w:
        return re.search(rf"(?:^|\W){re.escape(w)}(?:\W|$)", n) is not None
    if w in toks:
        return True
    if len(w) < 4:
        return False
    cyr = any("Ѐ" <= c <= "ӿ" for c in w)
    for t in toks:
        if cyr:
            k = 0
            while k < min(len(t), len(w)) and t[k] == w[k]:
                k += 1
            if k >= max(4, len(w) - 2) and abs(len(t) - len(w)) <= 3:
                return True
        elif len(w) >= 5 and t.startswith(w) and len(t) - len(w) <= 3:
            return True
    return False


DECADE_WORDS = {norm(k): v for k, v in {
    "الستينات": 1960, "السبعينات": 1970, "الثمانينات": 1980, "التسعينات": 1990, "الالفينات": 2000, "الألفينات": 2000,
    "шестидесят": 1960, "семидесят": 1970, "восьмидесят": 1980, "девяност": 1990, "нулевых": 2000, "нулевые": 2000,
    "六十年代": 1960, "七十年代": 1970, "八十年代": 1980, "九十年代": 1990, "90-е": 1990, "80-е": 1980, "90-х": 1990, "80-х": 1980,
    "90-an": 1990, "80-an": 1980, "1990-an": 1990, "2000-an": 2000, "2010-an": 2010, "1990 के दशक": 1990, "2000 के दशक": 2000,
    "1980 के दशक": 1980, "miaka ya 1990": 1990, "miaka ya 2000": 2000, "miaka ya 1980": 1980, "90'lar": 1990, "80'ler": 1980,
}.items()}
COVID_PHRASES = {k: [norm(x) for x in v] for k, v in {
    "pre_covid": ["pre-pandemic", "pre pandemic", "before covid", "pre-covid", "pre covid", "prepandemia", "pré-pandemia",
                  "pre-pandemia", "pré-covid", "avant la pandémie", "avant covid", "antes de la pandemia", "antes da pandemia",
                  "antes del covid", "до пандемии", "до ковида", "قبل كوفيد", "قبل الجائحة", "قبل كورونا", "疫情前", "新冠前",
                  "महामारी से पहले", "कोविड से पहले", "kabla ya covid", "kabla ya janga", "sebelum pandemi", "sebelum covid",
                  "pandemi öncesi", "pandemiden önce", "covid öncesi"],
    "post_covid": ["post-pandemic", "post pandemic", "after covid", "since covid", "post-covid", "après la pandémie",
                   "depuis la pandémie", "après covid", "después de la pandemia", "desde la pandemia", "pós-pandemia",
                   "após a pandemia", "после пандемии", "после ковида", "بعد كوفيد", "بعد الجائحة", "بعد كورونا", "منذ جائحة",
                   "疫情后", "疫情以来", "新冠以来", "महामारी के बाद", "कोविड के बाद", "baada ya covid", "baada ya janga",
                   "setelah pandemi", "pasca pandemi", "pascapandemi", "pandemi sonrası", "pandemiden sonra", "covid sonrası"],
}.items()}


def _edit_distance(a: str, b: str, limit: int) -> int:
    """Levenshtein distance, stopping early once every path exceeds `limit`."""
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i] + [0] * len(b)
        for j, cb in enumerate(b, 1):
            cur[j] = min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (ca != cb))
        if min(cur) > limit:
            return limit + 1
        prev = cur
    return prev[-1]


def _is_cjk(s: str) -> bool:
    return any("一" <= c <= "鿿" for c in s)


# Letters plus combining marks (Devanagari vowel signs, Arabic diacritics), so words are not split inside.
_MARKS = "̀-ͯ҃-҉֑-ֽؐ-ًؚ-ٰٟۖ-ۭऀ-ःऺ-ॏ॑-ॗॢॣঁ-ঃ়-ৗ"
_CH = rf"(?:[^\W_]|[{_MARKS}])"
TOKEN = re.compile(rf"{_CH}+(?:[-'’]{_CH}+)*", re.U)
AR_PREFIX = ("وال", "بال", "لل", "ال", "و", "ب", "ل", "ف")

# Standard English abbreviations for groupings; these decide between candidate groups for the same span.
ABBREV = {"mena": "CME", "ssa": "SSF", "lac": "LCN", "eap": "EAS", "eca": "ECS", "sar": "SAS", "ldc": "LDC", "ldcs": "LDC",
          "lmic": "LMC", "lmics": "LMC", "lic": "LIC", "lics": "LIC", "hic": "HIC", "hics": "HIC", "umic": "UMC", "umics": "UMC",
          "eu": "EUU", "oecd": "OED", "sids": "SST", "hipc": "HPC", "menaap": "MEA", "asean": "ASEAN", "gcc": "GCC",
          "ecowas": "ECOWAS", "sadc": "SADC", "eac": "EAC", "brics": "BRICS", "g7": "G7"}
# Everyday English names that map to the nearest grouping.
EXTRA_GROUP_NAMES = {"latin america": "LCN", "europe": "ECS", "east asia": "EAS", "middle east": "CME", "west africa": "AFW",
                     "east africa": "AFE", "southern africa": "AFE", "central africa": "AFW", "small island states": "SST",
                     "small island developing states": "SST", "low-income countries": "LIC", "middle-income countries": "MIC",
                     "high-income countries": "HIC", "least developed countries": "LDC", "emerging markets": "LMY"}
# English country variants absent from the official names.
COUNTRY_ALIASES = {"turkey": "TUR", "ivory coast": "CIV", "czech republic": "CZE", "swaziland": "SWZ", "burma": "MMR",
                   "cape verde": "CPV", "east timor": "TLS", "macedonia": "MKD", "drc": "COD", "dr congo": "COD",
                   "congo-kinshasa": "COD", "congo-brazzaville": "COG", "laos": "LAO", "syria": "SYR", "russia": "RUS",
                   "iran": "IRN", "egypt": "EGY", "yemen": "YEM", "south korea": "KOR", "north korea": "PRK", "vietnam": "VNM",
                   "gambia": "GMB", "bahamas": "BHS", "venezuela": "VEN", "kyrgyzstan": "KGZ", "slovakia": "SVK",
                   "micronesia": "FSM", "st lucia": "LCA", "saint lucia": "LCA", "brunei": "BRN", "palestine": "PSE",
                   "west bank": "PSE", "gaza": "PSE", "hong kong": "HKG", "macau": "MAC", "macao": "MAC", "uk": "GBR",
                   "britain": "GBR", "great britain": "GBR", "usa": "USA", "united states": "USA", "uae": "ARE",
                   "png": "PNG", "kosovo": "XKX", "bosnia": "BIH", "trinidad": "TTO", "st kitts": "KNA", "st vincent": "VCT"}


def _stem(tok: str) -> str:
    """Crude suffix trim for Russian case endings (index and query side)."""
    return tok[: max(3, len(tok) - 2)] if any("Ѐ" <= c <= "ӿ" for c in tok) else tok


def _query_stem(tok: str) -> str:
    """Query side only: also drop Turkish suffixes written after an apostrophe (Kenya'da, Türkiye'deki)."""
    return _stem(re.split(r"['’]", tok)[0])


class Understander:
    def __init__(self, geo: dict | None = None, lex: dict | None = None):
        self.geo = geo or json.loads(GEO.read_text())
        self.lex = lex or json.loads(LEX.read_text())
        self.countries = self.geo["countries"]
        self.groups = self.geo["groups"]
        self.names: dict[tuple, list[dict]] = {}  # token tuple -> candidate entities
        self.stems: dict[tuple, list[dict]] = {}  # stemmed token tuple -> candidates (fallback)
        self.cjk: list[tuple[str, dict]] = []
        stop = set(STOP_PLACES)
        common = COMMON if COMMON.exists() else _RES / "common_words.json"  # frequent ordinary words in searches
        if common.exists():
            stop.update(set(json.loads(common.read_text())) - ALLOW_PLACES)
        for lang in self.lex.values():  # time words such as COVID terms are never places
            for k in ("pre_covid", "post_covid", "latest", "recent"):
                for w in lang.get("time", {}).get(k, []):
                    stop.update(TOKEN.findall(norm(w)))
        for lang in self.lex.values():
            for sec in ("group", "datatype", "frequency", "scope"):
                for words in lang.get(sec, {}).values():
                    stop.update(norm(w) for w in words)
        for iso3, c in self.countries.items():
            for lang, ns in c["names"].items():
                for n in ns:
                    self._add(n, {"kind": "country", "iso3": iso3, "name": c["name"]}, stop_ok=True)
        from .sources import _ALIASES  # project aliases
        for alias, canon in _ALIASES.items():
            iso3 = self._iso3_by_name(canon)
            if iso3:
                self._add(alias, {"kind": "country", "iso3": iso3, "name": self.countries[iso3]["name"]}, stop_ok=True)
        for alias, iso3 in COUNTRY_ALIASES.items():
            if iso3 in self.countries:
                self._add(alias, {"kind": "country", "iso3": iso3, "name": self.countries[iso3]["name"]}, stop_ok=True)
        for lang in self.lex.values():  # per-language search aliases (synth.aliases), including inflected forms
            for iso3, ns in lang.get("country_aliases", {}).items():
                if iso3 in self.countries:
                    for n in ns:
                        self._add(n, {"kind": "country", "iso3": iso3, "name": self.countries[iso3]["name"]}, stop_ok=True)
        for code, g in self.groups.items():
            ent = {"kind": "group", "code": code, "name": g["name"]}
            self._add(g["name"], ent, stop_ok=True)
            for lang in self.lex.values():
                for n in lang.get("groupings", {}).get(g["name"], []) + lang.get("group_aliases", {}).get(code, []):
                    self._add(n, ent, stop_ok=True)
        for name, code in EXTRA_GROUP_NAMES.items():
            if code in self.groups:
                self._add(name, {"kind": "group", "code": code, "name": self.groups[code]["name"]}, stop_ok=True)
        for lang in self.lex.values():
            for key, words in lang.get("scope", {}).items():
                for w in words:
                    self._add(w, {"kind": "scope", "name": key}, stop_ok=True)
        for p in self.geo["places"]:
            ent = {"kind": p["kind"], "name": p["name"], "iso3": p["iso3"], "admin1": p.get("admin1", ""), "population": p.get("population", 0)}
            for n in {p["name"], p["ascii"], *p.get("alt", [])}:
                if len(n) >= 4 or _is_cjk(n):
                    self._add(n, ent, stop=stop)
        self.cyr: dict[int, list[tuple[tuple, list]]] = {}  # Cyrillic names by word count, countries and groups first
        rankk = {"country": 0, "group": 1, "scope": 2, "admin1": 3, "city": 4}
        items = [(k, v) for k, v in self.names.items() if any("Ѐ" <= c <= "ӿ" for t in k for c in t)]
        items.sort(key=lambda kv: min(rankk[c["kind"]] for c in kv[1]))
        for k, v in items:
            if all(c["kind"] in ("admin1", "city") for c in v) and len("".join(k)) < 5:
                continue
            self.cyr.setdefault(len(k), []).append((k, v))
        official = {tuple(TOKEN.findall(norm(n))) for c in self.countries.values() for ns in c["names"].values() for n in ns}
        for k, v in list(self.names.items()):  # an informal one-word country alias that names a place elsewhere: the place wins
            if len(k) == 1 and k not in official and any(c["kind"] == "country" for c in v):
                ctry = {c["iso3"] for c in v if c["kind"] == "country"}
                if any(c["kind"] in ("admin1", "city") and c["iso3"] not in ctry for c in v):
                    self.names[k] = [c for c in v if c["kind"] != "country"]
        cw = Path("data/geo/common_tokens.json") if Path("data/geo/common_tokens.json").exists() else _RES / "common_tokens.json"
        self.common_words = set(json.loads(cw.read_text())) if cw.exists() else set()
        self.latin_countries: dict[tuple, list[tuple[str, list]]] = {}  # (word count, first letter) -> country names
        for k, v in self.names.items():
            if all(t.isascii() for t in k) and len(" ".join(k)) >= 6 and any(c["kind"] == "country" for c in v):
                self.latin_countries.setdefault((len(k), k[0][0]), []).append((" ".join(k), [c for c in v if c["kind"] == "country"]))
        self._time_words()

    # ------------------------------------------------------------------ index
    def _add(self, name: str, ent: dict, stop: set | None = None, stop_ok: bool = False) -> None:
        n = norm(name.strip())
        if not n or (not stop_ok and stop is not None and n in stop):
            return
        if _is_cjk(n):
            self.cjk.append((n, ent))
            return
        key = tuple(TOKEN.findall(n))
        if key:
            lst = self.names.setdefault(key, [])
            if ent not in lst:
                lst.append(ent)
            sk = tuple(_stem(t) for t in key)  # fallback for inflected forms; only Cyrillic names change
            if sk != key and len("".join(sk)) >= 3:
                lst = self.stems.setdefault(sk, [])
                if ent not in lst:
                    lst.append(ent)

    def _cyrillic(self, key: tuple) -> list[dict] | None:
        """Russian case forms: each word shares with the name word a prefix of at least len-2 letters (at least 4
        letters for one-word names), so Кении matches Кения and Казахстане matches Казахстан, but данные does not
        match Дания."""
        def pref(a: str, b: str) -> int:
            k = 0
            while k < min(len(a), len(b)) and a[k] == b[k]:
                k += 1
            return k

        best, best_score = None, None
        for name_key, cands in self.cyr.get(len(key), []):
            only_places = all(c["kind"] in ("admin1", "city") for c in cands)
            need = [(3 if len(w) <= 4 or len(key) > 1 else 4) for w in name_key]
            need = [max(n, len(w) - 2) if len(w) > 4 else n for n, w in zip(need, name_key)]
            if only_places:  # cities and provinces: near-exact forms only (туризма is not Турин)
                need = [max(n, len(w) - 1) for n, w in zip(need, name_key)]
            slack = 1 if only_places else 2 if any(len(w) <= 4 for w in name_key) else 3  # перепис is not Перу
            if only_places and any(len(w) < 5 for w in name_key):
                continue
            if all(pref(q, w) >= n and len(q) <= len(w) + slack for q, w, n in zip(key, name_key, need)):
                score = sum(pref(q, w) - abs(len(q) - len(w)) for q, w in zip(key, name_key))
                if best_score is None or score > best_score:
                    best, best_score = cands, score
        return best

    def _typo(self, key: tuple) -> list[dict] | None:
        """Countries spelled with one wrong, missing or extra letter (two for names of 10+ letters), same first letter:
        indonesi, kazanakhstan, phillipines. Names under 6 letters are matched exactly only."""
        s = " ".join(key)
        if len(s) < 6 or any(t in self.common_words for t in key):  # real words (malaria) are not misspelt countries
            return None
        for name, cands in self.latin_countries.get((len(key), s[0]), []):
            limit = 2 if len(name) >= 10 else 1
            if abs(len(name) - len(s)) <= limit and _edit_distance(s, name, limit) <= limit:
                return cands
        return None

    def _iso3_by_name(self, name: str) -> str | None:
        n = norm(name)
        for iso3, c in self.countries.items():
            if any(norm(x) == n for x in c["names"].get("English", [])):
                return iso3
        return None

    def _time_words(self) -> None:
        t: dict[str, set] = {}
        for lang in self.lex.values():
            for k, words in lang.get("time", {}).items():
                t.setdefault(k, set()).update(norm(w) for w in words if w)
        self.tw = {k: sorted(v, key=len, reverse=True) for k, v in t.items()}
        self.lex_flat: dict[str, dict[str, set]] = {}
        for sec in ("frequency", "group", "datatype"):
            d: dict[str, set] = {}
            for lang in self.lex.values():
                for k, words in lang.get(sec, {}).items():
                    d.setdefault(k, set()).update(norm(w) for w in words if w and len(norm(w)) >= 3)
            self.lex_flat[sec] = d

    # ------------------------------------------------------------------ parsing
    def places(self, text: str) -> list[dict]:
        n = norm(text)
        toks = [(m.group(0), m.start(), m.end()) for m in TOKEN.finditer(n)]
        found, used = [], set()
        for size in range(min(7, len(toks)), 0, -1):  # longest match first
            for i in range(len(toks) - size + 1):
                if any(j in used for j in range(i, i + size)):
                    continue
                key = tuple(t for t, _, _ in toks[i : i + size])
                qk = tuple(re.split(r"['’]", t)[0] for t in key)  # Turkish: Kenya'da -> kenya
                cands = self.names.get(key) or (self.names.get(qk) if qk != key and len("".join(qk)) >= 4 else None)
                if not cands and size <= 3 and any("Ѐ" <= c <= "ӿ" for t in key for c in t):
                    cands = self._cyrillic(key)
                if not cands and size == 1:
                    w = key[0]
                    for p in AR_PREFIX:  # Arabic clitics written together with the name
                        if w.startswith(p) and len(w) - len(p) >= 3 and (cands := self.names.get((w[len(p):],))):
                            break
                if not cands and size <= 3 and all(t.isascii() for t in key):
                    cands = self._typo(key)
                if cands:
                    found.append({"span": n[toks[i][1] : toks[i + size - 1][2]], "cands": cands})
                    used.update(range(i, i + size))
        for name, ent in self.cjk:  # Chinese: no spaces between words
            if name in n and not any(name in f["span"] or f["span"] in name for f in found if _is_cjk(f["span"])):
                found.append({"span": name, "cands": [ent]})
        return self._resolve(found)

    def _resolve(self, found: list[dict]) -> list[dict]:
        rank = {"country": 0, "group": 1, "scope": 2, "admin1": 3, "city": 4}
        countries = {c["iso3"] for f in found for c in f["cands"] if c["kind"] == "country"}
        out = []
        for f in found:
            span = norm(f["span"])

            def group_pref(c):  # standard abbreviation, then own English name, then the smaller group
                if c["kind"] != "group":
                    return (0, 0, 0)
                g = self.groups[c["code"]]
                return (0 if ABBREV.get(span) == c["code"] else 1, 0 if norm(g["name"]) == span else 1, len(g["members"]))

            cands = sorted(f["cands"], key=lambda c: (rank[c["kind"]], group_pref(c), -c.get("population", 0)))
            top = cands[0]
            if top["kind"] in ("admin1", "city"):
                within = [c for c in cands if c.get("iso3") in countries]
                same_kind = [c for c in cands if c["kind"] == top["kind"]]
                if within:
                    top = within[0]
                elif len({c["iso3"] for c in same_kind}) > 1:
                    top = {**top, "ambiguous": sorted({self.countries[c["iso3"]]["name"] for c in same_kind})[:4]}
            out.append({**top, "span": f["span"]})
        seen, uniq = set(), []
        for p in out:  # one entry per entity
            key = (p["kind"], p.get("iso3"), p.get("code"), p["name"])
            if key not in seen:
                seen.add(key)
                uniq.append(p)
        return uniq

    def time(self, text: str) -> dict | None:
        n = norm(text)
        tw = self.tw
        alt = lambda k: "|".join(re.escape(w) for w in tw.get(k, []) if w)  # noqa: E731
        m = re.search(rf"({YEAR})\s*(?:-|–|—|/|{alt('between')})\s*({YEAR}|\d{{2}})(?!\d)", n)
        if m:
            a, b = int(m.group(1)), m.group(2)
            b = int(b) if len(b) == 4 else int(str(a)[:2] + b)
            if b >= a:
                return {"type": "range", "start": a, "end": b, "span": m.group(0)}
        m = re.search(r"(?:\b|^)((?:19[5-9]|20[0-3])0)\s*(?:'?s\b|年代|'?lar\b|'?ler\b|-?е\b|-?х\b)", n) or \
            re.search(rf"(?:anos|años|annees|décennie|decada|década)\s+(?:19|20)?(\d0)\b", n)
        if m:
            d = m.group(1)
            d = int(d) if len(d) == 4 else (1900 + int(d) if int(d) >= 50 else 2000 + int(d))
            return {"type": "decade", "start": d, "end": d + 9, "span": m.group(0)}
        for kind in ("since", "before"):
            if tw.get(kind):
                m = re.search(rf"(?:^|\W)({alt(kind)})\s+(?:(?:el|le|de|du|of|the|year|ano|año|année|mwaka|yil|tahun|سنة|عام|года?|साल)\s+)?({YEAR})", n)
                if m:
                    y = int(m.group(2))
                    return ({"type": "since", "start": y, "end": CURRENT_YEAR, "span": m.group(0).strip()} if kind == "since"
                            else {"type": "before", "start": None, "end": y, "span": m.group(0).strip()})
        m = re.search(rf"(?:^|\W)({alt('last')})\s+(\d{{1,2}})\s+({alt('years_word')})", n) if tw.get("last") and tw.get("years_word") else None
        if m:
            k = int(m.group(2))
            return {"type": "last", "start": CURRENT_YEAR - k, "end": CURRENT_YEAR, "span": m.group(0).strip()}
        years = re.findall(rf"(?<!\d)({YEAR})(?!\d)", n)
        if len(years) >= 2:
            ys = sorted(int(y) for y in years)
            return {"type": "years", "start": ys[0], "end": ys[-1], "list": ys, "span": " ".join(years)}
        if years:
            return {"type": "year", "start": int(years[0]), "end": int(years[0]), "span": years[0]}
        for word, d in DECADE_WORDS.items():
            if word in n:
                return {"type": "decade", "start": d, "end": d + 9, "span": word}
        toks = _variants(n)
        for kind, label in (("pre_covid", "before COVID-19"), ("post_covid", "since COVID-19")):
            for w in list(COVID_PHRASES[kind]) + tw.get(kind, []):
                if (len(w) >= 4 or _is_cjk(w)) and (w in n if (" " in w or "-" in w or _is_cjk(w)) else _has(n, toks, w)):
                    return ({"type": "before", "start": None, "end": 2019, "span": w, "label": label} if kind == "pre_covid"
                            else {"type": "since", "start": 2020, "end": CURRENT_YEAR, "span": w, "label": label})
        for kind in ("latest", "recent"):
            for w in tw.get(kind, []):
                if (len(w) >= 4 or (_is_cjk(w) and len(w) >= 2)) and _has(n, toks, w):
                    return {"type": kind, "start": CURRENT_YEAR - (3 if kind == "latest" else 5), "end": CURRENT_YEAR, "span": w}
        return None

    def _lexmatch(self, sec: str, text: str) -> list[str]:
        n, out = norm(text), []
        toks = _variants(n)
        for key, words in self.lex_flat[sec].items():
            if any(_has(n, toks, w) for w in words):
                out.append(key)
        return out

    def parse(self, query: str, facets: dict | None = None) -> dict:
        p = {"places": [], "time": None, "frequency": None, "groups": [], "datatype": None, "scope": None}
        for pl in self.places(query):
            if pl["kind"] == "scope":
                p["scope"] = pl["name"]
            else:
                p["places"].append({**pl, "source": "query"})
        t = self.time(query)
        if t:
            p["time"] = {**t, "source": "query"}
        fr = self._lexmatch("frequency", query)
        if fr:
            p["frequency"] = {"value": fr[0], "source": "query"}
        p["groups"] = [{"value": g, "source": "query"} for g in self._lexmatch("group", query)]
        dt = self._lexmatch("datatype", query)
        if dt:
            p["datatype"] = {"value": dt[0], "source": "query"}
        for key, val in (facets or {}).items():  # interface facets, in the first rounds' filter vocabulary
            if key == "countries":
                for c in val if isinstance(val, list) else [val]:
                    iso3 = self._iso3_by_name(c) or next((e["iso3"] for e in self.names.get(tuple(TOKEN.findall(norm(c))), []) if e["kind"] == "country"), None)
                    same = next((x for x in p["places"] if x.get("iso3") == iso3 and x["kind"] == "country"), None)
                    if same:  # the facet repeats a country already typed in the search
                        same["source"] = "query+facet"
                    elif iso3:
                        p["places"].append({"kind": "country", "iso3": iso3, "name": self.countries[iso3]["name"], "span": c, "source": "facet"})
            elif key == "years":
                t = self.time(str(val))
                if t:
                    q = p.get("time")
                    same = q and q.get("start") == t.get("start") and q.get("end") == t.get("end")
                    p["time"] = {**t, "source": "query+facet" if same else "facet"}
            elif key == "frequency":
                p["frequency"] = {"value": str(val).lower(), "source": "facet"}
            elif key in ("sex", "area", "age_group"):
                g = {"female": "female", "male": "male", "rural": "rural", "urban": "urban"}.get(str(val).lower())
                g = g or ("children" if key == "age_group" and re.search(r"0|5|child|month", str(val)) else "youth" if key == "age_group" and re.search(r"1[5-9]|2[0-4]", str(val)) else None)
                if g and not any(x["value"] == g for x in p["groups"]):
                    p["groups"].append({"value": g, "source": "facet"})
        return p

    # ------------------------------------------------------------------ rendering of the parse
    def describe(self, p: dict) -> str:
        parts = []
        for pl in p["places"]:
            if pl["kind"] == "group":
                d = f"place={pl['name']} (group of {len(self.groups[pl['code']]['members'])} countries)"
            elif pl["kind"] == "country":
                d = f"place={pl['name']} (country)"
            else:
                where = " or ".join(pl["ambiguous"]) if pl.get("ambiguous") else self.countries[pl["iso3"]]["name"]
                d = f"place={pl['name']} ({'city' if pl['kind'] == 'city' else 'province or state'} in {where})"
            parts.append(f"{d} [{pl['source']}]")
        if p.get("scope"):
            parts.append(f"place={'all countries' if p['scope'] == 'global' else 'developing countries'} [query]")
        t = p.get("time")
        if t:
            parts.append(f"time={time_label(t)} [{t['source']}]")
        if p.get("frequency"):
            parts.append(f"frequency={p['frequency']['value']} [{p['frequency']['source']}]")
        for g in p["groups"]:
            parts.append(f"group={g['value']} [{g['source']}]")
        if p.get("datatype"):
            parts.append(f"wants={p['datatype']['value']} [{p['datatype']['source']}]")
        return "; ".join(parts) or "no place, time or group constraints found"


def time_label(t: dict) -> str:
    k = t["type"]
    if k == "range":
        return f"{t['start']}-{t['end']}"
    if k == "decade":
        return f"{t['start']}s"
    if k == "since":
        return t.get("label") or f"{t['start']} or later"
    if k == "before":
        return t.get("label") or f"up to {t['end']}"
    if k == "last":
        return f"last {CURRENT_YEAR - t['start']} years ({t['start']}-{t['end']})"
    if k == "years":
        return ", ".join(map(str, t.get("list", [t["start"], t["end"]])))
    if k == "year":
        return str(t["start"])
    return "most recent data" if k == "latest" else "recent years"


# ---------------------------------------------------------------------- matching
@lru_cache(maxsize=1)
def _country_index() -> dict[str, str]:
    from .sources import _ALIASES
    geo = json.loads(GEO.read_text())
    idx = {}
    for iso3, c in geo["countries"].items():
        for n in c["names"].get("English", []) + [c["name"]]:
            idx[norm(n)] = iso3
    for alias, canon in _ALIASES.items():
        if norm(canon) in idx:
            idx[norm(alias)] = idx[norm(canon)]
    return idx


def _record_iso3(rec: dict) -> set[str]:
    idx = _country_index()
    cs = rec.get("countries") or []
    if isinstance(cs, str):  # some records store the list as text
        cs = re.findall(r"'([^']+)'", cs) or [cs]
    return {idx[norm(c)] for c in cs if norm(c) in idx}


def _record_text(rec: dict) -> str:
    return " ".join(str(rec.get(k) or "") for k in ("name", "definition", "concept", "abstract", "notes", "passage", "universe",
                                                    "geography", "unit_of_analysis", "unit", "tags", "topics"))


def check(rec: dict, p: dict, u: Understander) -> list[str]:
    out = []
    iso3s = _record_iso3(rec)
    n_cov = len(iso3s)
    text = _record_text(rec)
    ntext = norm(text)
    doc_region = rec.get("kind") == "document" and rec.get("countries") and not iso3s
    for pl in p["places"]:
        if pl["kind"] == "country":
            if iso3s and pl["iso3"] in iso3s:
                st = "covered"
            elif doc_region:
                st = f"document covers {', '.join(rec['countries'])[:60]}"
            elif not rec.get("countries"):
                st = "coverage not stated" + (" (named in the description)" if norm(pl["name"]) in ntext else "")
            elif rec.get("countries_truncated"):
                st = "cannot confirm (multi-country list not shown in full)"
            elif norm(pl["name"]) in ntext:
                st = "not in the coverage list but named in the description"
            else:
                st = "not covered" + (f" (item covers {', '.join(sorted(u.countries[c]['name'] for c in iso3s)[:3])})" if 0 < n_cov <= 3 else "")
            out.append(f"place {pl['name']}: {st}")
        elif pl["kind"] == "group":
            members = set(u.groups[pl["code"]]["members"])
            k = len(members & iso3s)
            if doc_region:
                st = f"document covers {', '.join(rec['countries'])[:60]}"
            elif not rec.get("countries"):
                st = "coverage not stated"
            elif k == len(members):
                st = f"covered (all {len(members)} countries)"
            elif k:
                st = f"partly covered ({k} of {len(members)} countries)"
            else:
                st = "not covered" + (f" (item covers {', '.join(sorted(u.countries[c]['name'] for c in iso3s)[:3])})" if 0 < n_cov <= 3 else "")
            out.append(f"place {pl['name']}: {st}")
        else:  # admin1 or city
            country = u.countries[pl["iso3"]]["name"] if not pl.get("ambiguous") else " or ".join(pl["ambiguous"])
            in_country = pl.get("ambiguous") is None and pl["iso3"] in iso3s
            named = norm(pl["name"]) in ntext
            sub = re.search(r"\b(district|province|regional|region-level|subnational|sub-national|admin ?[12]|county|municipal|state-level|departments?)\b", text, re.I)
            if named:
                st = "named in the metadata"
            elif in_country:
                st = f"{country} covered; " + ("subnational breakdown mentioned, this place not named" if sub else "subnational detail not stated")
            elif not rec.get("countries"):
                st = "coverage not stated"
            else:
                st = f"not covered ({country} not in coverage)"
            out.append(f"place {pl['name']}: {st}")
    if p.get("scope") == "global":
        out.append(f"place all countries: {'covered (' + str(n_cov) + ' countries)' if n_cov >= 150 else 'item covers ' + str(n_cov) + ' countries' if n_cov else 'coverage not stated'}")
    t = p.get("time")
    if t:
        lab = time_label(t)
        ys, ye = rec.get("year_start"), rec.get("year_end")
        if rec.get("kind") == "document":
            st = f"publication date {rec.get('date') or ys or 'not stated'}" + ("" if t["type"] in ("latest", "recent") else "; period studied not verifiable")
        elif not (ys and ye):
            st = "coverage not stated"
        elif t["type"] in ("latest", "recent"):
            st = f"latest year in the item {ye}" + (" (current)" if ye >= t["start"] else " (not recent)")
        else:
            a, b = t["start"] or ys, t["end"]
            point = ys == ye
            if ys <= a and b <= ye:
                st = f"covered ({ys}-{ye})" if not point else f"covered ({ys})"
            elif b < ys or a > ye:
                st = f"not covered ({ys}-{ye})" if not point else f"not covered (item year {ys})"
            else:
                st = f"partly covered ({ys}-{ye})"
        out.append(f"time {lab}: {st}")
    if p.get("frequency"):
        f, rf = p["frequency"]["value"], rec.get("frequency")
        out.append(f"frequency {f}: " + ("not stated" if not rf else "matches" if norm(rf).startswith(norm(f)[:5]) else f"differs ({rf})"))
    for g in p["groups"]:
        ok = re.search(GROUP_EVIDENCE[g["value"]], text, re.I)
        out.append(f"group {g['value']}: {'mentioned in the metadata' if ok else 'not mentioned in the metadata'}")
    if p.get("datatype"):
        want = DATATYPE_KIND[p["datatype"]["value"]]
        out.append(f"type {p['datatype']['value']}: " + ("matches" if want == rec.get("kind") else f"item is {KIND_LABEL.get(rec.get('kind'), rec.get('kind'))}"))
    return out
