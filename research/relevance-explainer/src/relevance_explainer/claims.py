"""Claim-carrying explanations, after Proof-Carrying Numbers (Solatorio 2025, arXiv 2509.06902).

The model first lists claims, each a value copied from one field of the record; the text then wraps every entity it
states in a reference to a claim. Verification happens here, outside the model, and fails closed: a claim is verified
only when its value occurs in the named field of the input, and a reference only when its words match its claim under
the policy below. Anything else is shown as unverified.

Output format, as the app and API see it:

    <verdict grade="7">partly_relevant</verdict>
    <claims><c cid="14" field="geography">Kenya</c><c cid="3" field="time">2015-2023</c></claims>
    <tldr>Covers <r cid="14">Kenya</r> for <r cid="3">2015 to 2023</r> ...</tldr>
    <exp>The indicator ... <r cid="14">Kenya</r> ...</exp>

`field` is a category shared by all catalogs (CATEGORY maps each record key to one), so "Coverage", "Country or
region" and "Geographic scope" are all geography; a claim verifies when its value occurs in any record field of its
category. Claim ids are drawn at random and the claims listed in random order, so an id says nothing about the field
or about where the text mentions it: the model must match a reference's words to a claim's value to choose the id.
The model itself writes a compact form with special tokens (to_model / from_model): <c><cid:14><f:geography>Kenya</c>
and <r><cid:14>Kenya</r>. `grade` (0 to 10) and the tldr are optional. Reference policy, in order: the same words after normalization (case,
Latin accents, dashes, spaces); the same numbers (year ranges in any language's wording); the same places (the
query-understanding gazetteer with each language's names and inflected forms). Words outside references are scanned
as well: a number absent from the input, or a place in neither the record nor the search, is reported as unverified.

`annotate` builds this format from an existing verdict and explanation by finding spans that match record fields, so
the existing teacher data can be converted without new generation.
"""
from __future__ import annotations

import re
import hashlib
import random
from dataclasses import dataclass, field
from functools import lru_cache

from .synth.understand import AR_PREFIX, TOKEN, Understander, _is_cjk, norm
from .tiny.numguard import RANGE, _forms, _join_thousands, supported_numbers

VERDICTS = ("relevant", "partly_relevant", "irrelevant")
QUERY_FIELDS = ("Respond in", "Search", "Facets", "Understood")  # describe the search, not the record
MAX_CLAIMS = 12
MAX_ID = 20  # ids are drawn from 1..MAX_ID
CATEGORY = {
    "Coverage": "geography", "Country or region": "geography", "Geographic scope": "geography",
    "Years": "time", "Survey years": "time", "Published": "time",
    "Frequency": "frequency", "Update frequency": "frequency",
    "Result": "title",
    "Source": "source", "Producer": "source", "Organization": "source", "Database": "source",
    "Item type": "type", "Document type": "type", "Type": "type", "Collection mode": "type",
    "Unit of analysis": "type", "Formats": "type",
    "Unit": "unit",
    "Tags": "topic", "Topics": "topic", "Concept": "topic", "Definition": "topic",
    "Universe": "population", "Variables": "variables",
    "Abstract": "text", "Description": "text", "Retrieved passage": "text", "Limitations": "text",
    "Checks": "check",
}
CATEGORIES = sorted(set(CATEGORY.values()) | {"other"})
_KEY = re.compile(r"(?:^|; )([A-Z][A-Za-z ()/]{1,30}): ")
_NUM = re.compile(r"\d+(?:[.,]\d+)*")
_GRADE = {"relevant": (6, 10), "partly_relevant": (3, 8), "irrelevant": (0, 4)}  # grades consistent with a verdict
_FAILED_CHECK = ("not covered", "differs")
# Fields that name places, in the order a place claim prefers them; long text fields come last. Places are not taken
# from names of organizations, sources or topics ("World" in "World Bank" is not a place).
GEO_FIELDS = ("Coverage", "Country or region", "Geographic scope", "Universe", "Checks", "Result")
TEXT_FIELDS = ("Abstract", "Description", "Retrieved passage", "Definition", "Limitations")
NOT_PLACES = {"того"}  # Russian "of that" (as in "кроме того"), spelled like the genitive of Togo


def _place_ok(p: dict) -> bool:
    """A gazetteer match that is a country or grouping, long enough not to be a short word, and one of the
    gazetteer's names as written. The gazetteer was built for searches, where "za" or a single Chinese character can
    be a place and typos are matched loosely (Russian "\u043f\u0435\u0440\u0438\u043e\u0434" would match Peru); in written text only exact names
    count."""
    span = p["span"]
    if span in NOT_PLACES:
        return False
    if any("\u3000" <= ch <= "\u9fff" for ch in span):
        return p["kind"] in ("country", "group") and len(span) >= 2
    return p["kind"] in ("country", "group") and len(span) >= 4 and tuple(TOKEN.findall(span)) in _u().names


_GENERIC = {"states", "countries", "small", "income", "and", "the", "of", "region", "area", "excluding", "high", "low",
            "middle", "upper", "lower", "&"}


def _group_words(text: str) -> set[str]:
    """Distinctive words of the groupings named in `text` ("Caribbean small states" and "Caribbean" share one)."""
    u = _u()
    return {w for p in _found(text) if p["kind"] == "group" and _place_ok(p)
            for w in TOKEN.findall(norm(p["name"])) if w not in _GENERIC}


@lru_cache(maxsize=1)
def _u() -> Understander:
    return Understander()


@lru_cache(maxsize=200_000)
def _found(text: str) -> tuple[dict, ...]:
    """Places named in written text: Understander.places without its loose matching for searches (typos, Cyrillic
    prefixes), which _place_ok would reject anyway and which takes most of the time. Exact names, Turkish suffixes
    after an apostrophe and Arabic clitics are kept. Cached, since record fields repeat across rows."""
    u = _u()
    n = norm(text)
    toks = [(m.group(0), m.start(), m.end()) for m in TOKEN.finditer(n)]
    found, used = [], set()
    for size in range(min(7, len(toks)), 0, -1):  # longest match first
        for i in range(len(toks) - size + 1):
            if any(j in used for j in range(i, i + size)):
                continue
            key = tuple(t for t, _, _ in toks[i : i + size])
            qk = tuple(re.split(r"['’]", t)[0] for t in key)
            cands = u.names.get(key) or (u.names.get(qk) if qk != key and len("".join(qk)) >= 4 else None)
            if not cands and size == 1:
                w = key[0]
                for p in AR_PREFIX:
                    if w.startswith(p) and len(w) - len(p) >= 3 and (cands := u.names.get((w[len(p):],))):
                        break
            if cands:
                found.append({"span": n[toks[i][1] : toks[i + size - 1][2]], "cands": cands})
                used.update(range(i, i + size))
    for name, ent in u.cjk:
        if name in n and not any(name in f["span"] or f["span"] in name for f in found if _is_cjk(f["span"])):
            found.append({"span": name, "cands": [ent]})
    return tuple(u._resolve(found))


def canon(s: str) -> str:
    """Normalized form for matching: lowercase, Latin accents dropped, dashes unified, spaces collapsed."""
    s = norm(s).replace("–", "-").replace("—", "-").replace("’", "'")
    return re.sub(r"\s+", " ", s).strip()


# ---------- the record ----------

def parse_record(input_text: str) -> list[tuple[str, str]]:
    """(field, value) pairs of the model input in order. A line holds one or more "Key: value" pairs separated by
    "; "; a line without a key continues the previous value."""
    out: list[list[str]] = []
    for line in input_text.split("\n"):
        keys = list(_KEY.finditer(line))
        if not keys or keys[0].start() != 0:
            if out:
                out[-1][1] += "\n" + line
            continue
        for i, m in enumerate(keys):
            end = keys[i + 1].start() if i + 1 < len(keys) else len(line)
            out.append([m.group(1), line[m.end() : end].strip()])
    return [(k, v) for k, v in out]


def record_fields(input_text: str) -> dict[str, str]:
    """Claimable fields: everything except the lines that describe the search."""
    d: dict[str, str] = {}
    for k, v in parse_record(input_text):
        if k not in QUERY_FIELDS:
            d[k] = d[k] + "; " + v if k in d else v
    return d


def _places(text: str) -> set[tuple]:
    return {(p["kind"], p.get("iso3"), p.get("code")) for p in _found(text) if _place_ok(p)}


def _numbers(text: str) -> list[str]:
    return [m.rstrip(".,") for m in _NUM.findall(_join_thousands(text))]


def _ranges(text: str) -> set[tuple[str, str]]:
    return {(m.group(1), m.group(3)) for m in RANGE.finditer(text)}


# ---------- parsing the structured output ----------

@dataclass
class Ref:
    claim: int
    text: str
    section: str
    ok: bool = False
    reason: str = ""


@dataclass
class Claim:
    id: int
    field: str
    value: str
    ok: bool = False
    reason: str = ""


@dataclass
class Checked:
    verdict: str
    grade: int | None
    claims: dict[int, Claim]
    refs: list[Ref]
    sections: dict[str, str]  # tldr / exp with references replaced by their words
    unverified: list[str] = field(default_factory=list)  # numbers and places outside references
    flags: list[str] = field(default_factory=list)  # consistency flags
    errors: list[str] = field(default_factory=list)  # format errors

    @property
    def verified_share(self) -> float | None:
        return sum(r.ok for r in self.refs) / len(self.refs) if self.refs else None

    @property
    def all_verified(self) -> bool:
        return not self.errors and all(c.ok for c in self.claims.values()) and all(r.ok for r in self.refs) and not self.unverified


_VERDICT = re.compile(r"<verdict(?: grade=\"(\d+)\")?>([a-z_]+)</verdict>")
_CLAIM = re.compile(r"<c cid=\"(\d+)\" field=\"([a-z]+)\">(.*?)</c>", re.S)
_REF = re.compile(r"<r cid=\"(\d+)\">(.*?)</r>", re.S)
_SECTION = {"tldr": re.compile(r"<tldr>(.*?)</tldr>", re.S), "exp": re.compile(r"<exp>(.*?)</exp>", re.S)}


def strip_refs(text: str) -> str:
    return _REF.sub(lambda m: m.group(2), text)


def parse(output: str) -> Checked:
    errors = []
    m = _VERDICT.search(output)
    verdict, grade = (m.group(2), int(m.group(1)) if m.group(1) else None) if m else ("invalid", None)
    if verdict not in VERDICTS:
        errors.append("no valid verdict")
        verdict = "invalid"
    cm = re.search(r"<claims>(.*?)</claims>", output, re.S)
    claims: dict[int, Claim] = {}
    for c in _CLAIM.finditer(cm.group(1) if cm else ""):
        if int(c.group(1)) in claims:
            errors.append(f"claim {c.group(1)} declared twice")
        claims[int(c.group(1))] = Claim(int(c.group(1)), c.group(2), c.group(3).strip())
    refs, sections = [], {}
    for name, rx in _SECTION.items():
        s = rx.search(output)
        if s is None:
            continue
        refs += [Ref(int(r.group(1)), r.group(2), name) for r in _REF.finditer(s.group(1))]
        sections[name] = strip_refs(s.group(1)).strip()
    if "exp" not in sections:
        errors.append("no explanation")
    return Checked(verdict, grade, claims, refs, sections, errors=errors)


# ---------- verification ----------

def category(key: str) -> str:
    return CATEGORY.get(key, "other")


def _claim_ok(c: Claim, fields: dict[str, str]) -> tuple[bool, str]:
    same = [v for k, v in fields.items() if category(k) == c.field]
    if not same:
        return False, f"no {c.field} field in the record"
    if not c.value:
        return False, "empty value"
    for fv in same:
        if canon(c.value) in canon(fv) or canon(_join_thousands(c.value)) in canon(_join_thousands(fv)):
            return True, ""
    return False, f"value not in any {c.field} field"


def _ref_ok(text: str, c: Claim) -> tuple[bool, str]:
    s, v = canon(text), canon(c.value)
    if len(s) >= 2 and (s == v or (len(s) >= 3 and re.search(rf"(?<!\w){re.escape(s)}(?!\w)", v))
                        or (len(v) >= 3 and re.search(rf"(?<!\w){re.escape(v)}(?!\w)", s))):
        return True, "words"
    nums = _numbers(text)
    if nums:  # every number is one of the claim's (in any written form); a stated range is the claim's range
        have = set().union(*(_forms(n) for n in _numbers(c.value)))
        rs = _ranges(text)
        if all(n in have for n in nums) and rs <= _ranges(c.value):
            return True, "numbers"
        return False, "numbers differ"
    ps, pv = _places(text), _places(c.value)
    if ps and ps <= pv:
        return True, "place"
    return False, "words differ"


def _outside_refs(text: str, record: str, search: str) -> list[str]:
    """Numbers absent from the input and places in neither the record nor the search, in text outside references."""
    bare = _REF.sub(" ", text)
    have = supported_numbers(record + "\n" + search)
    out = [n for n in _numbers(bare) if n not in have]
    known, groups = _places(record) | _places(search), _group_words(record + "\n" + search)
    known_names = " | ".join(norm(p["name"]) for t in (record, search) for p in _found(t) if _place_ok(p))
    for p in _found(bare):
        if not _place_ok(p) or (p["kind"], p.get("iso3"), p.get("code")) in known:
            continue
        if p["kind"] == "group" and set(TOKEN.findall(norm(p["name"]))) & groups:  # the same grouping, named differently
            continue
        if re.search(rf"\b{re.escape(norm(p['name']))}\b", known_names):  # part of a known name ("Macao SAR, China")
            continue
        out.append(p["span"])
    return out


def verify(output: str, input_text: str) -> Checked:
    ck = parse(output)
    fields = record_fields(input_text)
    for c in ck.claims.values():
        c.ok, c.reason = _claim_ok(c, fields)
    for r in ck.refs:
        c = ck.claims.get(r.claim)
        if c is None:
            r.ok, r.reason = False, f"no claim {r.claim}"
        elif not c.ok:
            r.ok, r.reason = False, f"claim {r.claim} not verified"
        else:
            r.ok, r.reason = _ref_ok(r.text, c)
            r.reason = "" if r.ok else r.reason
    search = "\n".join(f"{k}: {v}" for k, v in parse_record(input_text) if k in QUERY_FIELDS)
    record = "\n".join(f"{k}: {v}" for k, v in fields.items())
    raw = {name: rx.search(output) for name, rx in _SECTION.items()}
    for name, m in raw.items():
        if m:
            ck.unverified += _outside_refs(m.group(1), record, search)
    ck.flags = consistency(ck, fields)
    return ck


def consistency(ck: Checked, fields: dict[str, str]) -> list[str]:
    """Mechanical contradictions between the verdict, its grade and the deterministic checks of the input."""
    flags = []
    if ck.grade is not None and ck.verdict in _GRADE:
        lo, hi = _GRADE[ck.verdict]
        if not lo <= ck.grade <= hi:
            flags.append(f"grade {ck.grade} unlikely for {ck.verdict}")
    failed = [c.strip() for c in fields.get("Checks", "").split("|") if any(f in c for f in _FAILED_CHECK)]
    if ck.verdict == "relevant" and failed:
        flags.append("relevant although a check failed: " + "; ".join(failed))
    return flags


# ---------- converting existing explanations ----------

def _char_map(text: str) -> tuple[str, list[int]]:
    """norm() applied character by character, with the original index of every normalized character."""
    out, idx = [], []
    for i, ch in enumerate(text):
        n = norm(ch)
        out.append(n)
        idx += [i] * len(n)
    return "".join(out), idx


def _find_word(text: str, span: str) -> int:
    """Start of `span` in `text` as a whole word (anywhere for Chinese, which has no spaces), or -1."""
    if any("\u3000" <= ch <= "\u9fff" for ch in span):
        return text.find(span)
    m = re.search(rf"(?<!\w){re.escape(span)}(?!\w)", text)
    return m.start() if m else -1


def _spans(narrative: str, fields: dict[str, str]) -> list[tuple[int, int, str, str]]:
    """Candidate (start, end, field, value) spans of the narrative that state a record value."""
    cands = []
    order = sorted(fields, key=lambda k: (k == "Description", k == "Checks"))  # specific fields first
    # year ranges and numbers
    for m in RANGE.finditer(narrative):
        y1, y2 = m.group(1), m.group(3)
        for k in order:
            fm = next((r for r in RANGE.finditer(fields[k]) if (r.group(1), r.group(3)) == (y1, y2)), None)
            if fm:
                cands.append((m.start(), m.end(), k, fm.group(0)))
                break
    for m in _NUM.finditer(narrative):
        tok = m.group(0).rstrip(".,")
        if len(tok) < 2:
            continue
        for k in order:
            fv = _join_thousands(fields[k])
            hit = next((f for f in _forms(tok) if re.search(rf"(?<![\d.,]){re.escape(f)}(?![\d])", fv)), None)
            if hit:
                cands.append((m.start(), m.start() + len(tok), k, hit))
                break
    # places
    nn, idx = _char_map(narrative)
    porder = [k for k in GEO_FIELDS if k in fields] + [k for k in TEXT_FIELDS if k in fields]
    field_places = {k: [q for q in _found(fields[k]) if _place_ok(q)] for k in porder}
    for p in _found(narrative):
        if not _place_ok(p):
            continue
        key = (p["kind"], p.get("iso3"), p.get("code"))
        i = _find_word(nn, p["span"])
        if i < 0:
            continue
        for k in porder:
            fp = next((q for q in field_places[k] if (q["kind"], q.get("iso3"), q.get("code")) == key), None)
            if fp is None:
                continue
            fn, fidx = _char_map(fields[k])
            j = _find_word(fn, fp["span"])
            if j < 0:
                continue
            value = fields[k][fidx[j] : fidx[j + len(fp["span"]) - 1] + 1]
            cands.append((idx[i], idx[i + len(p["span"]) - 1] + 1, k, value))
            break
    # names and titles copied word for word (items of at least 4 characters)
    for k in order:
        if k in ("Description", "Checks"):
            continue
        for item in {x.strip() for x in re.split(r"[;,]\s*|\n", fields[k]) if len(x.strip()) >= 4}:
            for m in re.finditer(rf"(?<!\w){re.escape(item)}(?!\w)", narrative, re.I):
                cands.append((m.start(), m.end(), k, item))
    # longest first, no overlaps
    cands.sort(key=lambda c: (-(c[1] - c[0]), c[0]))
    kept: list[tuple[int, int, str, str]] = []
    for c in cands:
        if all(c[1] <= o[0] or c[0] >= o[1] for o in kept):
            kept.append(c)
    return sorted(kept)


def annotate(verdict: str, narrative: str, input_text: str, grade: int | None = None, tldr: str | None = None,
             seed: int | None = None) -> str:
    """The structured output for an existing verdict and explanation (and optional tldr): claims for the record
    values the text states, and references around them. Removing the tags gives back the original texts. Claim ids
    are drawn at random from 1..MAX_ID and the claims listed in random order; the default seed is derived from the
    texts, so a conversion is reproducible, and another seed gives another permutation of the same example."""
    fields = record_fields(input_text)
    if seed is None:
        seed = int(hashlib.sha256((input_text + "\0" + narrative).encode()).hexdigest()[:12], 16)
    rng = random.Random(seed)
    spans = {name: _spans(text, fields) for name, text in (("tldr", tldr or ""), ("exp", narrative))}
    keys: list[tuple[str, str]] = []
    for name in ("tldr", "exp"):
        for _, _, k, v in spans[name]:
            key = (category(k), v)
            if key not in keys and len(keys) < MAX_CLAIMS:
                keys.append(key)
    ids = dict(zip(keys, rng.sample(range(1, MAX_ID + 1), len(keys))))

    def tag(name: str, text: str) -> str:
        out, pos = [], 0
        for s, e, k, v in spans[name]:
            i = ids.get((category(k), v))
            if i is None:
                continue
            out += [text[pos:s], f'<r cid="{i}">{text[s:e]}</r>']
            pos = e
        return "".join(out) + text[pos:]

    t = tag("tldr", tldr) if tldr else None
    x = tag("exp", narrative)
    g = f' grade="{grade}"' if grade is not None else ""
    order = list(ids.items())
    rng.shuffle(order)
    claims = "".join(f'<c cid="{i}" field="{f}">{v}</c>' for (f, v), i in order)
    parts = [f"<verdict{g}>{verdict}</verdict>", f"<claims>{claims}</claims>"]
    if t is not None:
        parts.append(f"<tldr>{t}</tldr>")
    parts.append(f"<exp>{x}</exp>")
    return "\n".join(parts)


# ---------- the model's compact form ----------
# The model writes line-based text so that the verdict stays the first generated word (the verdict probabilities,
# the trainer's verdict weight and the app all read the first line), with one special token per tag, id, field and
# grade. Special tokens look like <|name|>; an input never contains "<|" (sanitize_input), so a record cannot inject
# a tag. Layout (B has no claims line; the A format is the plain "verdict\nexplanation"):
#
#   partly_relevant
#   <|g:7|>
#   <|claims|><|c|><|cid:14|><|f:geography|>Kenya<|/c|>...<|/claims|>
#   <|tldr|>Covers <|r|><|cid:14|>Kenya<|/r|> ...<|/tldr|>
#   <|exp|>The indicator ...<|/exp|>

SPECIAL_TOKENS = (["<|claims|>", "<|/claims|>", "<|c|>", "<|/c|>", "<|r|>", "<|/r|>", "<|tldr|>", "<|/tldr|>", "<|exp|>",
                   "<|/exp|>"] + [f"<|cid:{i}|>" for i in range(1, MAX_ID + 1)] + [f"<|f:{c}|>" for c in CATEGORIES]
                  + [f"<|g:{i}|>" for i in range(11)])
_PAIRS = [("<claims>", "<|claims|>"), ("</claims>", "<|/claims|>"), ("</c>", "<|/c|>"), ("</r>", "<|/r|>"),
          ("<tldr>", "<|tldr|>"), ("</tldr>", "<|/tldr|>"), ("<exp>", "<|exp|>"), ("</exp>", "<|/exp|>")]


def sanitize_input(text: str) -> str:
    """Inputs never contain the special-token prefix, so record text cannot open a tag."""
    return text.replace("<|", "< |")


def strip_model_tags(text: str) -> str:
    """Model text without any special token (for number and script checks on the words alone)."""
    return re.sub(r"<\|[^|<>]{1,24}\|>", "", text)  # no space: Chinese has none between words


def to_model(text: str) -> str:
    """Attribute form to the model's line-based form."""
    m = _VERDICT.search(text)
    verdict, grade = (m.group(2), m.group(1)) if m else ("invalid", None)
    body = text[m.end():] if m else text
    body = re.sub(r'<c cid="(\d+)" field="([a-z]+)">', r"<|c|><|cid:\1|><|f:\2|>", body)
    body = re.sub(r'<r cid="(\d+)">', r"<|r|><|cid:\1|>", body)
    for a, b in _PAIRS:
        body = body.replace(a, b)
    lines = [verdict] + ([f"<|g:{grade}|>"] if grade is not None else []) + [x for x in body.strip().split("\n") if x]
    return "\n".join(lines)


def from_model(text: str) -> str:
    """The model's line-based form back to the attribute form (for verify and the app)."""
    first, _, body = text.partition("\n")
    g = re.match(r"\s*<\|g:(\d+)\|>\n?", body)
    grade = f' grade="{g.group(1)}"' if g else ""
    body = body[g.end():] if g else body
    body = re.sub(r"<\|c\|><\|cid:(\d+)\|><\|f:([a-z]+)\|>", r'<c cid="\1" field="\2">', body)
    body = re.sub(r"<\|r\|><\|cid:(\d+)\|>", r'<r cid="\1">', body)
    for a, b in _PAIRS:
        body = body.replace(b, a)
    return f"<verdict{grade}>{first.strip()}</verdict>\n{body.strip()}"


def plain_output(text: str) -> str:
    """"verdict\nexplanation" in words only, from any format: the explanation section without tags when the text has
    one, else everything after the first line. Used for the judge and the text metrics, so all formats are compared
    on the same words."""
    first, _, rest = text.partition("\n")
    m = re.search(r"<\|exp\|>(.*?)(?:<\|/exp\|>|$)", rest, re.S)
    body = m.group(1) if m else rest
    body = re.sub(r"<\|r\|><\|cid:\d+\|>(.*?)<\|/r\|>", r"\1", body, flags=re.S)
    return f"{first.strip()}\n{re.sub(r' {2,}', ' ', strip_model_tags(body)).strip()}"


def tldr_of(text: str) -> str | None:
    """The tl;dr in words only, or None when the text has none."""
    m = re.search(r"<\|tldr\|>(.*?)(?:<\|/tldr\|>|$)", text, re.S)
    if not m:
        return None
    body = re.sub(r"<\|r\|><\|cid:\d+\|>(.*?)<\|/r\|>", r"\1", m.group(1), flags=re.S)
    return re.sub(r" {2,}", " ", strip_model_tags(body)).strip()
