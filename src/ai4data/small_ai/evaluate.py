"""Scores against real reference labels, and the gap report for the feedback loop.

Only labels that occur in the reference are measured, because a label
the reference never uses cannot be checked.

Multi-label tasks: a document gets the labels whose score is at or above
the threshold, at most `top` of them. The scores are micro precision,
recall, and F1 (pooled over document-label pairs) and macro F1 (the mean
of the per-label F1).

Single-label tasks: a text gets its highest-scoring code. The scores are
the accuracy at each level of the hierarchy (for example the first one,
two, three, and four digits of an ISCO-08 code), the accuracy of the top
three codes, macro F1 over the measured codes, and the accuracy when
only the texts with the most confident predictions are coded
automatically (50 and 80 percent of them).

The gap report lists, per measured label, the support, precision,
recall, F1, and the number of times the label was assigned, together with
the most frequent confusions: for a single-label task, pairs of the
reference code and the predicted code; for a multi-label task, pairs of a
reference label and a label assigned without reference.
"""

from __future__ import annotations

from collections import Counter

import numpy as np
import pandas as pd

from .task import Task


def choose(
    scores: dict[str, float], threshold: float, k: int | None = None
) -> list[str]:
    ranked = sorted(scores.items(), key=lambda item: (-item[1], item[0]))
    chosen = [label for label, s in ranked if s >= threshold]
    return chosen[:k] if k else chosen


def joined(gold: pd.DataFrame, pred: list[dict]) -> pd.DataFrame:
    frame = pd.DataFrame(pred)
    frame["id"] = frame["id"].astype(str)
    gold = gold.assign(id=gold["id"].astype(str))
    return gold[["id", "labels"]].merge(frame, on="id", how="inner")


def multi_label(
    gold: pd.DataFrame, pred: list[dict], threshold: float, top: int | None = 5
) -> dict:
    from sklearn.metrics import f1_score, precision_score, recall_score

    data = joined(gold, pred)
    measured = sorted({x for ls in data["labels"] for x in ls})
    truth = np.array([[x in ls for x in measured] for ls in data["labels"]], dtype=int)
    chosen = [set(choose(s, threshold, top)) for s in data["scores"]]
    guess = np.array([[x in c for x in measured] for c in chosen], dtype=int)
    confusions: Counter = Counter()
    for ls, c in zip(data["labels"], chosen):
        for wrong in c - set(ls):
            for right in set(ls) - c:
                confusions[(right, wrong)] += 1
    per = per_label(truth, guess, measured)
    return {
        "documents": len(data),
        "measured_labels": len(measured),
        "threshold": threshold,
        "top": top,
        "micro_precision": round(
            float(precision_score(truth, guess, average="micro", zero_division=0)), 4
        ),
        "micro_recall": round(
            float(recall_score(truth, guess, average="micro", zero_division=0)), 4
        ),
        "micro_f1": round(
            float(f1_score(truth, guess, average="micro", zero_division=0)), 4
        ),
        "macro_f1": round(
            float(f1_score(truth, guess, average="macro", zero_division=0)), 4
        ),
        "per_label": per,
        "confusions": [
            {"reference": a, "predicted": b, "count": n}
            for (a, b), n in confusions.most_common(25)
        ],
    }


def sweep(
    gold: pd.DataFrame,
    pred: list[dict],
    top: int | None = 5,
    grid=(0.02, 0.03, 0.05, 0.075, 0.1, 0.15, 0.2, 0.3, 0.4, 0.5),
) -> list[tuple[float, float]]:
    return [(t, multi_label(gold, pred, t, top)["micro_f1"]) for t in grid]


def single_label(task: Task, gold: pd.DataFrame, pred: list[dict]) -> dict:
    from sklearn.metrics import f1_score

    data = joined(gold, pred)
    truth = [ls[0] for ls in data["labels"]]
    ranked = [sorted(s.items(), key=lambda kv: -kv[1]) for s in data["scores"]]
    guess = [r[0][0] for r in ranked]
    confidence = np.array([r[0][1] for r in ranked])
    top3 = [[c for c, _ in r[:3]] for r in ranked]
    measured = sorted(set(truth))
    correct = np.array([a == b for a, b in zip(truth, guess)])
    levels = {}
    for level in task.levels or [max(len(c) for c in truth)]:
        levels[str(level)] = round(
            float(np.mean([a[:level] == b[:level] for a, b in zip(truth, guess)])), 4
        )
    coverage = {}
    order = np.argsort(-confidence)
    for share in (0.5, 0.8):
        k = max(1, round(share * len(order)))
        coverage[str(share)] = round(float(correct[order[:k]].mean()), 4)
    labels = sorted(set(truth) | set(guess))
    truth_m = np.array([[t == x for x in measured] for t in truth], dtype=int)
    guess_m = np.array([[g == x for x in measured] for g in guess], dtype=int)
    confusions = Counter((a, b) for a, b in zip(truth, guess) if a != b)
    return {
        "texts": len(data),
        "measured_labels": len(measured),
        "predicted_labels": len(set(guess)),
        "accuracy": round(float(correct.mean()), 4),
        "accuracy_by_level": levels,
        "top3_accuracy": round(
            float(np.mean([t in c for t, c in zip(truth, top3)])), 4
        ),
        "macro_f1": round(
            float(
                f1_score(
                    truth, guess, labels=measured, average="macro", zero_division=0
                )
            ),
            4,
        ),
        "accuracy_at_coverage": coverage,
        "per_label": per_label(truth_m, guess_m, measured),
        "confusions": [
            {"reference": a, "predicted": b, "count": n}
            for (a, b), n in confusions.most_common(25)
        ],
        "never_measured_predictions": int(sum(g not in set(measured) for g in guess)),
        "_labels": len(labels),
    }


def per_label(truth: np.ndarray, guess: np.ndarray, measured: list[str]) -> list[dict]:
    out = []
    for j, label in enumerate(measured):
        t, g = truth[:, j], guess[:, j]
        tp = int((t & g).sum())
        support, assigned = int(t.sum()), int(g.sum())
        p = tp / assigned if assigned else 0.0
        r = tp / support if support else 0.0
        f = 2 * p * r / (p + r) if p + r else 0.0
        out.append(
            {
                "label": label,
                "support": support,
                "assigned": assigned,
                "precision": round(p, 3),
                "recall": round(r, 3),
                "f1": round(f, 3),
            }
        )
    return out


def evaluate(
    task: Task,
    gold: pd.DataFrame,
    pred: list[dict],
    threshold: float | None = None,
    top: int | None = 5,
) -> dict:
    if task.multi_label:
        return multi_label(gold, pred, threshold if threshold is not None else 0.2, top)
    return single_label(task, gold, pred)


def headline(task: Task, result: dict) -> float:
    """The one score the loop compares rounds on: micro F1, or accuracy at the leaf level."""
    return result["micro_f1"] if task.multi_label else result["accuracy"]


def summary(task: Task, result: dict) -> str:
    if task.multi_label:
        return (
            f"documents {result['documents']}  measured labels {result['measured_labels']}  threshold {result['threshold']}\n"
            f"micro P {result['micro_precision']:.3f}  R {result['micro_recall']:.3f}  F1 {result['micro_f1']:.3f}  "
            f"macro F1 {result['macro_f1']:.3f}"
        )
    levels = "  ".join(
        f"{k}-digit {v:.3f}" for k, v in result["accuracy_by_level"].items()
    )
    cov = "  ".join(
        f"{float(k):.0%} coded {v:.3f}"
        for k, v in result["accuracy_at_coverage"].items()
    )
    return (
        f"texts {result['texts']}  measured codes {result['measured_labels']}\n"
        f"accuracy {result['accuracy']:.3f}  top-3 {result['top3_accuracy']:.3f}  macro F1 {result['macro_f1']:.3f}\n"
        f"by level: {levels}\nmost confident: {cov}"
    )


def gap_report(
    task: Task,
    result: dict,
    train_counts: Counter,
    agreement: dict[str, float] | None = None,
    limit: int = 40,
) -> dict:
    """What the orchestrator reads: weak labels, over-assigned labels, confusions, thin labels."""
    per = pd.DataFrame(result["per_label"])
    name = {x: task.labels[x].name for x in task.labels}
    per["name"] = per["label"].map(name)
    per["train_examples"] = per["label"].map(lambda x: train_counts.get(x, 0))
    if agreement:
        per["labeller_agreement"] = per["label"].map(lambda x: agreement.get(x))
    measured = per[per["support"] >= int(task.reference.get("min_support", 3))]
    weak = measured.sort_values(["f1", "support"], ascending=[True, False]).head(limit)
    over = measured.assign(
        excess=measured["assigned"] - measured["support"]
    ).sort_values("excess", ascending=False)
    over = over[over["excess"] > 0].head(15)
    unmeasured = [x for x in task.leaves if x not in set(per["label"])]
    thin = sorted(task.leaves, key=lambda x: train_counts.get(x, 0))[:20]
    return {
        "headline": headline(task, result),
        "summary": summary(task, result),
        "weak_labels": weak.drop(columns=[c for c in ("excess",) if c in weak]).to_dict(
            "records"
        ),
        "over_assigned": over.drop(columns=["excess"]).to_dict("records"),
        "confusions": [
            dict(
                c,
                reference_name=name.get(c["reference"]),
                predicted_name=name.get(c["predicted"]),
            )
            for c in result["confusions"]
        ],
        "thin_labels": [
            {"label": x, "name": name[x], "train_examples": train_counts.get(x, 0)}
            for x in thin
        ],
        "labels_not_in_feedback_set": len(unmeasured),
    }
