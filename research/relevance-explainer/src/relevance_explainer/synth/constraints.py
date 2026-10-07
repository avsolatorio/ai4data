"""Deterministic constraints added in render version 3: thresholds (money per day, age bands), exclusions (a place or
population group the search excludes) and named periods. The challenge suite showed the student fails on these when
it must infer them from text (threshold 45%, exclusion 50%, indirect time 62%), so the preprocessor states them and the
checks compare them with the record, as it already does for places and years.

All matching is on `understand.norm` text (lowercase, Latin accents removed). Cue lists cover the eleven languages.
"""
from __future__ import annotations

import re

NUM = r"(\d+(?:[.,]\d{1,2})?)"
DAY = (r"(?:a|per|each|/)\s*day|daily|a\s*dia|al\s*dia|por\s*dia|par\s*jour|/\s*jour|par\s*personne\s*et\s*par\s*jour|"
       r"в\s*день|в\s*сутки|في\s*اليوم|يوميا|يومياً|每天|每日|一天|per\s*hari|sehari|/\s*hari|gunluk|günlük|gunde|günde|"
       r"kwa\s*siku|प्रतिदिन|प्रति\s*दिन|रोज़|रोज")
CUR_PRE = r"(?:\$|us\$|usd\s*|ppp\s*\$|\$\s*ppp\s*)"
CUR_POST = r"(?:\s*(?:\$|usd|dollars?|dolares|dólares|dollars|долл\w*|دولار\w*|美元|dolar|dola|डॉलर))"
MONEY_DAY = [
    re.compile(rf"{CUR_PRE}\s*{NUM}\s*(?:{CUR_POST})?\s*(?:{DAY})"),
    re.compile(rf"(?<![\d.,]){NUM}{CUR_POST}\s*(?:{DAY})"),
]
WORD_NUM = {"one": 1, "two": 2, "three": 3, "four": 4, "five": 5, "deux": 2, "trois": 3, "dos": 2, "tres": 3, "dois": 2, "duas": 2,
            "два": 2, "три": 3, "dua": 2, "tiga": 3, "iki": 2, "uc": 3, "üç": 3, "mbili": 2, "tatu": 3}
MONEY_WORD = re.compile(rf"(?<!\w)({'|'.join(WORD_NUM)})\s+(?:dollars?|dolares|dólares|долл\w*|dolar|dola)\s*(?:{DAY})")

AGE_UNDER = [re.compile(p) for p in (
    r"(?:under|below|less than|younger than)\s*(?:age\s*)?(\d{1,2})(?!\d)", r"under[- ](five|5)s?\b", r"(\d{1,2})\s*(?:and|or)\s*(?:under|younger)",
    r"(?:moins de|de moins de)\s*(\d{1,2})\s*ans", r"(?:menores de|menos de)\s*(\d{1,2})\s*(?:anos|años)", r"menores de\s*(\d{1,2})\b",
    r"(?:младше|до)\s*(\d{1,2})\s*лет", r"(\d{1,2})\s*岁以下", r"(\d{1,2})岁以下", r"(?:di bawah|dibawah|kurang dari)\s*(\d{1,2})\s*tahun",
    r"(\d{1,2})\s*yas\s*alti", r"(\d{1,2})\s*yaş\s*altı", r"chini ya (?:miaka\s*)?(\d{1,2})", r"(\d{1,2})\s*(?:वर्ष|साल)\s*से\s*कम",
    r"دون\s*(?:سن\s*)?(\d{1,2})", r"أقل من\s*(\d{1,2})\s*سن")]
AGE_OVER = [re.compile(p) for p in (
    r"(?:over|above|older than|aged over)\s*(?:age\s*)?(\d{2})(?!\d)", r"(\d{2})\s*(?:\+|and over|and older|or older|and above)",
    r"(?:plus de|de plus de)\s*(\d{2})\s*ans", r"(?:mayores de|mas de|más de|mais de)\s*(\d{2})\s*(?:anos|años)", r"старше\s*(\d{2})",
    r"(\d{2})\s*岁以上", r"(?:di atas|diatas|lebih dari)\s*(\d{2})\s*tahun", r"(\d{2})\s*(?:yas|yaş)\s*(?:ustu|üstü)",
    r"zaidi ya (?:miaka\s*)?(\d{2})", r"(\d{2})\s*(?:वर्ष|साल)\s*से\s*(?:अधिक|ज़्यादा|ऊपर)", r"فوق\s*(?:سن\s*)?(\d{2})")]
AGE_RANGE = re.compile(r"(?:aged?|ages|age group|entre|de|between|от|usia|umur|yas|yaş|miaka|आयु|عمر|年龄)?\s*(\d{1,2})\s*(?:-|–|to|a|à|y|e|и|до|hingga|sampai|ile|hadi|से|إلى|至|到)\s*(\d{1,2})\s*(?:years?|ans|años|anos|лет|岁|tahun|yas|yaş|miaka|वर्ष|साल|سنة|عاما)")

AGE_RANGE_PREFIXED = re.compile(r"(?:aged|ages|age group|age|edad(?:es)? de|idade de|usia|umur|возраст\w*|年龄)\s*(\d{1,2})\s*(?:-|–|to|a|à|до|hingga)\s*(\d{1,2})(?!\d)")

# Exclusion cues: before the excluded word (most languages) or after it (Turkish, Hindi postpositions). Only explicit
# exclusion words: "without", "sans", "sin", "без", "tanpa" and similar usually name a concept ("sans emploi" is
# unemployed, "без гражданства" stateless, "sin estudio ni empleo" NEET), so they are not cues.
EXCL_BEFORE = re.compile(r"(?:^|\W)(?:excluding|except(?: for)?|other than|apart from|but not|not including|"
                         r"sauf|hors|excepte|a l'exception de|autre que|excepto|salvo|excluyendo|aparte de|"
                         r"exceto|excluindo|кроме|исключая|за исключением|除了|不包括|不含|باستثناء|ما عدا|عدا|"
                         r"kecuali|selain|isipokuwa|सिवाय|not)\s*(?:the |la |le |les |el |los |las |o |a |os |as )?$")
EXCL_AFTER = re.compile(r"^\s*(?:haric|hariç|disinda|dışında|haricinde|hariçinde|को छोड़कर|के अलावा)")


def _num(s: str) -> float:
    return float(s.replace(",", "."))


def money_per_day(text: str) -> list[float]:
    vals = [_num(m.group(1)) for p in MONEY_DAY for m in p.finditer(text)]
    vals += [float(WORD_NUM[m.group(1)]) for m in MONEY_WORD.finditer(text)]
    return sorted({v for v in vals if 0.5 <= v <= 50})


def ages(text: str) -> list[tuple]:
    """Age constraints as (kind, lo, hi): ("under", None, 5), ("over", 65, None), ("range", 15, 24)."""
    out = []
    for p in AGE_UNDER:
        for m in p.finditer(text):
            v = 5 if m.group(1) in ("five",) else int(m.group(1))
            if 1 <= v <= 30:
                out.append(("under", None, v))
    for p in AGE_OVER:
        for m in p.finditer(text):
            v = int(m.group(1))
            if 40 <= v <= 90:
                out.append(("over", v, None))
    for m in list(AGE_RANGE.finditer(text)) + list(AGE_RANGE_PREFIXED.finditer(text)):
        a, b = int(m.group(1)), int(m.group(2))
        if 0 <= a < b <= 99:
            out.append(("range", a, b))
    if re.search(r"0\s*-\s*59\s*months|under-?five|under 5|u5", text):
        out.append(("under", None, 5))
    return list(dict.fromkeys(out))


def thresholds(nquery: str) -> list[dict]:
    out = [{"kind": "money_per_day", "value": v} for v in money_per_day(nquery)]
    out += [{"kind": f"age_{k}", "lo": lo, "hi": hi} for k, lo, hi in ages(nquery)]
    return out


def label(t: dict) -> str:
    if t["kind"] == "money_per_day":
        return f"${t['value']:.2f} a day"
    if t["kind"] == "age_under":
        return f"age under {t['hi']}"
    if t["kind"] == "age_over":
        return f"age {t['lo']} and over"
    return f"age {t['lo']}-{t['hi']}"


def excluded(nquery: str, span: str) -> bool:
    """The matched span is preceded (or, for postpositions, followed) by an exclusion cue."""
    i = nquery.find(span)
    if i < 0:
        return False
    before, after = nquery[max(0, i - 40):i], nquery[i + len(span):i + len(span) + 20]
    return bool(EXCL_BEFORE.search(before) or EXCL_AFTER.search(after))


EVENTS = [  # (pattern on norm text, time dict)
    (r"(?:before|prior to|avant|antes de|antes da|до|قبل|之前|sebelum|oncesi|öncesi|kabla ya|से पहले)\s*(?:the |la |el |a )?(?:2008|global financial|financial)\s*crisis|(?:2008|financial) crisis\s*(?:öncesi|oncesi|से पहले)",
     {"type": "before", "start": None, "end": 2007, "label": "before the 2008 financial crisis"}),
    (r"(?:after|since|apres|depuis|despues de|desde|apos|после|بعد|以来|之后|setelah|sonrasi|sonrası|baada ya|के बाद)\s*(?:the |la |el |a )?(?:2008|global financial|financial)\s*crisis",
     {"type": "since", "start": 2009, "end": 2026, "label": "since the 2008 financial crisis"}),
    (r"(?:global financial crisis|crise financiere|crisis financiera|crise financeira|финансов\w* кризис|الأزمة المالية|金融危机|krisis keuangan|finansal kriz|mgogoro wa kifedha|वित्तीय संकट)",
     {"type": "range", "start": 2008, "end": 2009, "label": "the 2008-2009 financial crisis"}),
    (r"ebola", {"type": "range", "start": 2014, "end": 2016, "label": "the 2014-2016 Ebola outbreak"}),
    (r"arab spring|printemps arabe|primavera arabe|арабск\w* весн|الربيع العربي|阿拉伯之春", {"type": "range", "start": 2011, "end": 2012, "label": "the Arab Spring (2011-2012)"}),
    (r"(?:last|past|previous|dernier\w*|ultim\w*|последн\w*|过去|terakhir|son|iliyopita|पिछले)\s*(?:decade|decennie|decada|década|десятилети\w*|十年|dekade|on yil|on yıl|muongo|दशक)",
     {"type": "last", "start": 2016, "end": 2026, "label": "the last 10 years (2016-2026)"}),
]
FISCAL = re.compile(r"(?:^|\W)fy\s*'?(\d{2}|\d{4})(?:\s*/\s*(\d{2}))?(?!\d)")


def named_period(nquery: str) -> dict | None:
    for pat, t in EVENTS:
        m = re.search(pat, nquery)
        if m:
            return {**t, "span": m.group(0).strip()}
    m = FISCAL.search(nquery)
    if m:
        y = int(m.group(1)) if len(m.group(1)) == 4 else 2000 + int(m.group(1))
        return {"type": "range" if m.group(2) else "year", "start": y, "end": y + 1 if m.group(2) else y,
                "span": m.group(0).strip(), "label": f"fiscal year {m.group(0).strip().upper()}"}
    return None


def record_thresholds(text: str) -> dict:
    """Money-per-day lines and age bands stated in the record's metadata."""
    from .understand import norm
    n = norm(text)
    money = sorted(set(money_per_day(n)) | {_num(m.group(1)) for m in re.finditer(rf"{CUR_PRE}\s*{NUM}\s*(?:a day|per day|daily|\(20\d\d ppp\))", n)}
                   | {_num(m.group(1)) for m in re.finditer(r"\$\s*(\d+[.,]\d{2})", n)})
    return {"money": [v for v in money if 0.5 <= v <= 50], "ages": ages(n)}


def check_threshold(t: dict, rt: dict) -> str:
    lab = label(t)
    if t["kind"] == "money_per_day":
        if not rt["money"]:
            return f"threshold {lab}: no poverty line stated in the metadata"
        if any(abs(v - t["value"]) < 0.06 for v in rt["money"]):
            return f"threshold {lab}: matches"
        return f"threshold {lab}: differs (record uses {', '.join(f'${v:.2f} a day' for v in rt['money'][:3])})"
    if not rt["ages"]:
        return f"{lab}: age band not stated in the metadata"
    want = (t.get("lo"), t.get("hi"))
    for k, lo, hi in rt["ages"]:
        if (k, lo, hi) == (t["kind"][4:], *want):
            return f"{lab}: matches"
    have = ", ".join(label({"kind": f"age_{k}", "lo": lo, "hi": hi}) for k, lo, hi in rt["ages"][:2])
    return f"{lab}: differs (record covers {have})"


# ---------- units: what kind of number the search asks for, against what the record gives ----------
UNIT_QUERY = {
    "percent": r"percent\w*|percentage|share of|proportion|pourcentage|part des?|porcentaje|proporcion|porcentagem|"
               r"процент\w*|доля|百分比|比例|persentase|yuzde|yüzde|asilimia|प्रतिशत|نسبة|بالمئة",
    # explicit count words only: "berapa", "كم", "сколько", "多少", "combien" also mean "how much"
    "count": r"how many|number of|total number|count of|nombre de|numero de|número de|cuantos|cuántos|quantos|"
             r"количеств\w*|数量|人数|jumlah|sayisi|sayısı|idadi ya|कितने|संख्या|عدد",
    "per_capita": r"per capita|per person|per head|per inhabitant|par habitant|par personne|por habitante|por persona|"
                  r"por pessoa|на душу|на человека|人均|per kapita|kisi basina|kişi başına|kwa kila mtu|प्रति व्यक्ति|للفرد|لكل فرد",
    "currency": r"dollar value|value in dollars|in us\$|in usd|en dolares|en dólares|em dolares|в долларах|美元|dalam dolar|"
                r"dolar cinsinden|kwa dola|डॉलर में|بالدولار",
}
UNIT_LABEL = {"percent": "a percentage or rate", "count": "a count", "per_capita": "a per-person or per-population rate",
              "currency": "a money value"}


def unit_of_query(nquery: str) -> str | None:
    for k in ("per_capita", "currency", "percent", "count"):  # the more specific unit first ("porcentaje ... por habitante")
        pat = UNIT_QUERY[k]
        if re.search(rf"(?<!\w)(?:{pat})", nquery) or (k == "percent" and "%" in nquery):
            return k
    return None


def unit_of_record(rec: dict) -> str | None:
    from .understand import norm
    s = norm(" ".join(str(rec.get(k) or "") for k in ("unit", "name")))
    if re.search(r"per capita|per person|per 1,?000|per 100,?000|per 10,?000|per inhabitant", s):
        return "per_capita"
    if re.search(r"%|percent|share of|proportion|ratio", s):
        return "percent"
    if re.search(r"us\$|current \$|constant \$|lcu|dollars|usd|ppp \$", s):
        return "currency"
    if re.search(r"\bnumber\b|\bcount\b|\btotal\b|\bpeople\b|\bpersons\b|\bthousands?\b|\bmillions?\b", s):
        return "count"
    return None


def check_unit(want: str, rec: dict) -> str:
    have = unit_of_record(rec)
    if not have:
        return f"unit {UNIT_LABEL[want]}: unit not stated in the metadata"
    if have == want or {have, want} == {"percent", "per_capita"}:
        return f"unit {UNIT_LABEL[want]}: matches"
    return f"unit {UNIT_LABEL[want]}: differs (record gives {UNIT_LABEL[have]})"


# ---------- excluded terms that are not places or groups ("excluding agriculture", "without imputed values") ----------
_CUES = (r"excluding|except(?: for)?|other than|apart from|but not|not including|sauf|hors|excepte|a l'exception de|"
         r"autre que|excepto|salvo|excluyendo|aparte de|exceto|excluindo|кроме|исключая|за исключением|"
         r"باستثناء|ما عدا|kecuali|selain|isipokuwa")  # explicit cues only (see EXCL_BEFORE)
_ART = ("the", "la", "le", "les", "el", "los", "las", "o", "a", "os", "as", "l'")
EXCL_TERM = re.compile(rf"(?:^|\W)(?:{_CUES})\s+([^\W\d][\w'-]*(?:\s+[^\W\d][\w'-]*){{0,2}})")
_STOP_END = {"in", "for", "of", "the", "and", "or", "from", "since", "before", "after", "en", "de", "du", "des", "del", "da",
             "do", "y", "et", "e", "и", "в", "di", "na", "ya"}


EXCL_CJK = re.compile(r"(?:不含|不包括|除了|排除)([一-鿿]{1,8}?)(?:以外|之外|外|[）)，,。\s]|$)")
_W = r"[\wऀ-ॿ'-]"  # \w misses Devanagari vowel signs
EXCL_POST = re.compile(rf"((?:{_W}+\s+){{0,1}}{_W}+)\s+(?:haric|hariç|disinda|dışında|haricinde|को छोड़कर|के अलावा)")


def excluded_terms(nquery: str) -> list[str]:
    out = [m.group(1) for m in EXCL_CJK.finditer(nquery)] + [m.group(1) for m in EXCL_POST.finditer(nquery)]
    for m in EXCL_TERM.finditer(nquery):
        words = m.group(1).split()
        if words and words[0] in _ART:
            words = words[1:]
        for i, w in enumerate(words):  # stop at a connective ("excluding agriculture in Kenya" -> "agriculture")
            if w in _STOP_END:
                words = words[:i]
                break
        if words:
            out.append(" ".join(words))
    return list(dict.fromkeys(out))
