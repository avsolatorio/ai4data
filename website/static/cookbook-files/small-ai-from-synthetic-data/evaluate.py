"""Compare predicted document labels with reference labels.

Reads reference labels (JSON lines with id and labels) and predicted
scores (JSON lines with id and scores, as predict.py writes them). Only
the labels that occur in the reference file are measured, because a
label the reference never uses cannot be checked. A document gets the
labels whose score is at or above --threshold, at most --top of them.
Prints, over the measured labels:

    accuracy (all cells)   share of document-label cells predicted right,
                           shown to make the point that it is not a useful score
    micro precision, recall, F1   pooled over all document-label pairs
    macro F1                      the average of the per-label F1 scores

the per-label F1 for the five weakest and five strongest labels (among
labels with at least ten reference documents), and the three labels
assigned most often relative to the reference.
With --sweep, it prints micro F1 for thresholds from 0.02 to 0.5, so
that the threshold can be chosen on a calibration file.

Uses pandas and scikit-learn.

Usage:
    python evaluate.py real_test.jsonl scores.jsonl --threshold 0.2 --top 5
    python evaluate.py real_test.jsonl scores.jsonl --sweep

Exit status: 0 always.

What this does not check: the quality of the reference labels. Labels
taken from a catalogue were assigned for search, and a low precision can
then be an omission in the reference.
"""

from __future__ import annotations

import argparse
import sys

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from topic_scores import choose


def matrices(
    gold: pd.DataFrame, pred: pd.DataFrame, threshold: float, top: int | None
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    measured = sorted({label for labels in gold["labels"] for label in labels})
    joined = gold.merge(pred, on="id", how="inner")
    truth = np.array(
        [[label in labels for label in measured] for labels in joined["labels"]],
        dtype=int,
    )
    chosen = [set(choose(scores, threshold, top)) for scores in joined["scores"]]
    guess = np.array([[label in c for label in measured] for c in chosen], dtype=int)
    return truth, guess, measured


def report(truth: np.ndarray, guess: np.ndarray) -> dict[str, float]:
    return {
        "accuracy (all cells)": accuracy_score(truth.ravel(), guess.ravel()),
        "micro precision": precision_score(
            truth, guess, average="micro", zero_division=0
        ),
        "micro recall": recall_score(truth, guess, average="micro", zero_division=0),
        "micro F1": f1_score(truth, guess, average="micro", zero_division=0),
        "macro F1": f1_score(truth, guess, average="macro", zero_division=0),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("reference")
    parser.add_argument("scores")
    parser.add_argument("--threshold", type=float, default=0.2)
    parser.add_argument("--top", type=int, default=5)
    parser.add_argument("--sweep", action="store_true")
    args = parser.parse_args(argv)
    gold = pd.read_json(args.reference, lines=True, dtype={"id": str})[["id", "labels"]]
    pred = pd.read_json(args.scores, lines=True, dtype={"id": str})
    if args.sweep:
        for threshold in (0.02, 0.03, 0.05, 0.075, 0.1, 0.15, 0.2, 0.3, 0.4, 0.5):
            truth, guess, _ = matrices(gold, pred, threshold, args.top)
            print(
                f"threshold {threshold:<6g} micro F1 {report(truth, guess)['micro F1']:.3f}"
            )
        return 0
    truth, guess, measured = matrices(gold, pred, args.threshold, args.top)
    print(
        f"documents {len(truth)}  measured labels {len(measured)}  threshold {args.threshold}  top {args.top}"
    )
    for name, value in report(truth, guess).items():
        print(f"{name:<22}{value:.3f}")
    per_label = pd.Series(
        f1_score(truth, guess, average=None, zero_division=0), index=measured
    )
    support = pd.Series(truth.sum(axis=0), index=measured)
    per_label = per_label[support >= 10].sort_values()
    print("weakest: " + ", ".join(f"{k} {v:.2f}" for k, v in per_label.head(5).items()))
    print(
        "strongest: "
        + ", ".join(f"{k} {v:.2f}" for k, v in per_label.tail(5)[::-1].items())
    )
    assigned = pd.Series(guess.sum(axis=0), index=measured)
    excess = (assigned / support.clip(lower=1)).sort_values(ascending=False)
    print(
        "most over-assigned: "
        + ", ".join(
            f"{k} ({assigned[k]} assigned, {support[k]} in reference)"
            for k in excess.index[:3]
        )
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
