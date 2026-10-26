"""Turn sentence-level label probabilities into document topics.

Used by predict.py and evaluate.py, and importable on its own. A
classifier trained on single sentences reads a long document one
sentence at a time. These functions split a text into sentences, combine
the sentence probabilities into one score per label for the document,
and choose the document's labels.

    split_sentences(text)          sentences of at least four words
    aggregate(rows, how)           "mean" or "max" of the rows, per label
    choose(scores, threshold, k)   labels at or above the threshold, at most k

Uses numpy.

What this does not do: weigh sentences by position or importance. A
title or a summary sentence counts as much as any other.
"""

from __future__ import annotations

import re

import numpy as np

BOUNDARY = re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9\"'(])")


def split_sentences(text: str, min_words: int = 4) -> list[str]:
    parts = [p.strip() for p in BOUNDARY.split(" ".join(text.split()))]
    sentences = [p for p in parts if len(p.split()) >= min_words]
    return sentences or [text.strip()]


def aggregate(rows: np.ndarray, how: str = "mean") -> np.ndarray:
    rows = np.asarray(rows, dtype=float)
    if how == "mean":
        return rows.mean(axis=0)
    if how == "max":
        return rows.max(axis=0)
    raise ValueError(f"unknown aggregation: {how}")


def choose(
    scores: dict[str, float], threshold: float, k: int | None = None
) -> list[str]:
    ranked = sorted(scores.items(), key=lambda item: (-item[1], item[0]))
    chosen = [label for label, score in ranked if score >= threshold]
    return chosen[:k] if k else chosen
