"""Whether a request is in one of the model's languages.

The model was trained on searches and explanations in the eleven languages of synth.geo.LANG_CODES. Searches in other
languages were answered in a third language with garbled words (notebook 2026-10-16), so a request outside them is
withheld. Two tests:

- the response language ("Respond in:") must be one of the eleven;
- the search must not be in another language: by script (Japanese kana with Chinese characters, or letters of Korean,
  Thai and other scripts the model has no data in making up at least 30% of the letters), then by fastText language ID
  when data/geo/lid.176.ftz is present, on the lowercased search, for Latin-script searches of MIN_WORDS words or more
  and only when fastText is confident (FASTTEXT_MIN).

On the evaluation searches (6,396) and 50,000 training searches, all in the eleven languages, the rules refuse
2 and 12 of them (0.03% and 0.02%; place names such as Gjirokastër, short Indonesian and Portuguese keyword
searches read as Italian). Short German or Tagalog keyword searches still pass (notebook 2026-10-16).
"""
from __future__ import annotations

import re

from .synth.geo import LANG_CODES

SUPPORTED = set(LANG_CODES)
# fastText codes accepted besides the eleven: close languages that keyword searches in the eleven are often labelled as
ACCEPT = set(LANG_CODES.values()) | {"ms", "gl", "ca"}
FASTTEXT_MIN = 0.9
MIN_WORDS = 3
KANA = re.compile(r"[\u3040-\u30ff]")
HAN = re.compile(r"[\u4e00-\u9fff]")
NON_LATIN = re.compile(r"[\u0600-\u06ff\u0900-\u097f\u0400-\u04ff]")  # Arabic, Devanagari, Cyrillic: judged by script alone
# Scripts outside the eleven languages: Hiragana and Katakana, Hangul, Thai, Lao, Khmer, Myanmar, Ethiopic, Georgian,
# Armenian, Hebrew, Bengali, Gurmukhi, Gujarati, Tamil, Telugu, Kannada, Malayalam, Sinhala, Tibetan
OTHER_SCRIPT = re.compile(r"[぀-ヿ가-힯ᄀ-ᇿ฀-๿຀-໿ក-៿"
                          r"က-႟ሀ-፿Ⴀ-ჿ԰-֏֐-׿ঀ-৿਀-੿"
                          r"઀-૿஀-௿ఀ-౿ಀ-೿ഀ-ൿ඀-෿ༀ-࿿]")


def _field(text: str, name: str) -> str:
    m = re.search(rf"(?m)^{name}: (.*)$", text)
    return m.group(1).strip() if m else ""


def unsupported(text: str) -> str | None:
    """A reason when the rendered input's response language or search language is outside the model's languages."""
    lang = _field(text, "Respond in")
    if lang and lang not in SUPPORTED:
        return f"The model does not write explanations in {lang}."
    search = _field(text, "Search")
    letters = re.findall(r"[^\W\d_]", search)
    other = len(OTHER_SCRIPT.findall(search))
    if KANA.search(search) and HAN.search(search) or letters and other / len(letters) >= 0.3:
        return "The search is in a language the model does not support."
    if HAN.search(search) or NON_LATIN.search(search) or len(re.findall(r"[^\W\d_]+", search)) < MIN_WORDS:
        return None
    from .synth import realistic
    if not realistic.LID_PATH.exists():
        return None
    if realistic._LID is None:
        import fasttext
        realistic._LID = fasttext.load_model(str(realistic.LID_PATH))
    pred = realistic._LID.f.predict(search.lower().replace("\n", " "), 1, 0.0, "strict")
    if pred and pred[0][0] >= FASTTEXT_MIN and pred[0][1].replace("__label__", "") not in ACCEPT:
        return "The search is in a language the model does not support."
    return None
