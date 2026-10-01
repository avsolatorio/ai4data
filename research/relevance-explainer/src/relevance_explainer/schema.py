"""Structured match diagnostics: the input contract for the explainer model."""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from enum import Enum


class MatchLevel(str, Enum):
    EXACT = "exact"
    SYNONYM = "synonym"
    BROADER = "broader"
    NARROWER = "narrower"


@dataclass
class Correspondence:
    """A query concept paired with the record value that satisfies it."""

    query_concept: str
    record_value: str
    field: str  # e.g. "indicator", "sex", "age", "title", "description"
    level: MatchLevel = MatchLevel.EXACT


@dataclass
class MatchDiagnostics:
    query: str
    result_title: str
    similarity: float
    matched: list[Correspondence] = field(default_factory=list)
    missing: list[str] = field(default_factory=list)
    user_question: str | None = None

    def to_dict(self) -> dict:
        d = asdict(self)
        for m in d["matched"]:
            m["level"] = m["level"].value
        return d
