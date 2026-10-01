"""Template-based explainer. Reference point for any learned model."""
from __future__ import annotations

from .schema import Correspondence, MatchDiagnostics, MatchLevel

_STRONG = 0.75

_LEVEL_PHRASE = {
    MatchLevel.EXACT: "matches",
    MatchLevel.SYNONYM: "is covered by",
    MatchLevel.BROADER: "is covered by the broader concept",
    MatchLevel.NARROWER: "is covered by the narrower concept",
}


def _join(items: list[str]) -> str:
    if len(items) <= 1:
        return "".join(items)
    return ", ".join(items[:-1]) + " and " + items[-1]


def _describe(c: Correspondence) -> str:
    return f"'{c.query_concept}' {_LEVEL_PHRASE[c.level]} '{c.record_value}' ({c.field})"


def explain(d: MatchDiagnostics) -> str:
    if not d.matched:
        return (
            f"'{d.result_title}' has no matched concepts for this query. "
            "It appears only through weak semantic similarity."
        )
    if d.missing:
        verdict = "partial match"
        gap = f" It does not cover {_join(d.missing)}, so it only partially satisfies the query."
    elif d.similarity >= _STRONG:
        verdict, gap = "strong match", ""
    else:
        verdict, gap = "weak match", " Semantic similarity is low despite the concept overlap."
    body = "; ".join(_describe(c) for c in d.matched)
    return f"'{d.result_title}' is a {verdict}: {body}.{gap}"
