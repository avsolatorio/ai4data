"""Request plans: which labels each request to the generating model is about.

A balanced plan offers every label about equally often. Each set of
labels is drawn with probabilities that favour the labels offered least
so far: a label offered one time fewer than another is e (about 2.7)
times as likely to be drawn. Examples that already exist count as offers.
With `group="siblings"` (for hierarchical codes), each request starts
from one drawn label and adds labels with the same parent, so that the
writer has to tell close codes apart.

A targeted plan comes from the feedback loop: a list of items, each with
the labels to write about, a number of requests, a mode, and a note that
says why. The modes are

    positive   examples of the labels, as in a balanced plan
    contrast   examples near the boundary between the labels, each
               labelled with the one it belongs to (hard negatives for
               the others)
    background examples about none of the labels (multi-label tasks only)

Every request is a dict with request_id, labels, mode, and note.
"""

from __future__ import annotations

from collections import Counter

import numpy as np

MODES = ("positive", "contrast", "background")


def balanced(
    leaves: list[str],
    requests: int,
    per_request: int,
    existing: Counter | None = None,
    per_offer: float = 4.0,
    seed: int = 42,
    uniform: bool = False,
    siblings: dict[str, list[str]] | None = None,
    prefix: str = "req",
) -> list[dict]:
    rng = np.random.default_rng(seed)
    existing = existing or Counter()
    offered = np.array([existing.get(x, 0) / per_offer for x in leaves], dtype=float)
    index = {x: i for i, x in enumerate(leaves)}
    out = []
    for n in range(requests):
        weights = (
            np.ones(len(leaves)) if uniform else np.exp(-(offered - offered.min()))
        )
        if siblings is None:
            picked = list(
                rng.choice(
                    len(leaves),
                    size=per_request,
                    replace=False,
                    p=weights / weights.sum(),
                )
            )
        else:
            first = int(rng.choice(len(leaves), p=weights / weights.sum()))
            near = [index[s] for s in siblings.get(leaves[first], []) if s in index]
            near = sorted(near, key=lambda i: (offered[i], rng.random()))[
                : per_request - 1
            ]
            picked = [first, *near]
            if len(picked) < per_request:
                rest = np.array([i for i in range(len(leaves)) if i not in picked])
                w = weights[rest]
                more = rng.choice(
                    rest,
                    size=min(per_request - len(picked), len(rest)),
                    replace=False,
                    p=w / w.sum(),
                )
                picked += [int(i) for i in more]
        offered[picked] += 1
        out.append(
            {
                "request_id": f"{prefix}-{n:05d}",
                "labels": sorted(leaves[i] for i in picked),
                "mode": "positive",
                "note": "",
            }
        )
    return out


def targeted(items: list[dict], known: set[str], prefix: str) -> list[dict]:
    """Expand the items of a targeted plan into requests, after checking them."""
    out = []
    for k, item in enumerate(items):
        mode = item.get("mode", "positive")
        if mode not in MODES:
            raise ValueError(f"item {k}: unknown mode {mode!r}")
        labels = sorted(set(item.get("labels", [])))
        unknown = [x for x in labels if x not in known]
        if unknown:
            raise ValueError(f"item {k}: unknown labels {unknown}")
        if mode == "contrast" and len(labels) < 2:
            raise ValueError(f"item {k}: a contrast item needs at least two labels")
        if mode != "background" and not labels:
            raise ValueError(f"item {k}: no labels")
        for j in range(int(item.get("requests", 1))):
            out.append(
                {
                    "request_id": f"{prefix}-{k:03d}-{j:03d}",
                    "labels": labels if mode != "background" else [],
                    "mode": mode,
                    "note": str(item.get("note", ""))[:300],
                }
            )
    return out


def offers(plan: list[dict]) -> Counter:
    return Counter(x for request in plan for x in request["labels"])
