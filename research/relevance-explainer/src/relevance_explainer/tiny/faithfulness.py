"""Cheap faithfulness proxy: numbers and capitalized names in a narrative that do not appear in its input.

Not a full factuality check (it cannot see wrong claims built from supported words), but it catches invented
years, counts, places and titles, and it needs no model.
"""
from __future__ import annotations

import re
import unicodedata

_NUM = re.compile(r"\d[\d.,]*\d|\d")
_CAP = re.compile(r"(?<![.!?]\s)(?<!^)\b([A-Z][a-zA-ZÀ-ɏ]{3,})")


def _norm(s: str) -> str:
    return unicodedata.normalize("NFKD", s).encode("ascii", "ignore").decode().lower()


def unsupported(narrative: str, source: str) -> list[str]:
    """Numbers (2+ digits) and capitalized words (4+ letters) of the narrative that the input does not contain."""
    src = _norm(source)
    nums = {m.group(0).rstrip(".,") for m in _NUM.finditer(narrative) if len(re.sub(r"\D", "", m.group(0))) >= 2}
    caps = {m.group(1) for m in _CAP.finditer(narrative)}
    return sorted(t for t in nums | caps if _norm(t) not in src)


def rate(pairs: list[tuple[str, str]]) -> dict:
    """Share of narratives with at least one unsupported item, and mean count per narrative."""
    bad = [unsupported(n, s) for n, s in pairs]
    return {"n": len(pairs), "narratives_with_unsupported": sum(1 for b in bad if b) / max(len(bad), 1),
            "mean_unsupported": sum(len(b) for b in bad) / max(len(bad), 1)}
