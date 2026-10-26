"""Plan the requests to the generating model so that every label is covered.

Each request to the large model asks for sentences about a small set of
labels. If the sets were drawn uniformly, some labels would by chance be
offered far less often than others, and some combinations would never
occur. This script draws each set with probabilities that favour the
labels offered least so far: a label offered one time fewer than the
least offered label so far is e (about 2.7) times as likely to be drawn as
one offered once more. Examples that already exist (from an earlier run,
with --existing) count as offers, at --examples-per-offer examples per
offer. With --uniform, every label is equally likely at every draw, for
comparison. With --background N, the plan ends with N requests that have
no labels, which ask for sentences about none of the topics (see
generate_examples.py). Writes one JSON line per request
(request_id, labels) and prints how often the most and least offered
labels were offered.

Uses pandas and numpy.

Usage:
    python plan_requests.py labels.csv requests.jsonl --requests 1000 --per-request 8
    python plan_requests.py labels.csv requests.jsonl --existing examples.jsonl

Exit status: 0 when the plan is written.

What this does not do: make combinations meaningful. Two labels drawn
together may rarely co-occur in real text; the generating model is
asked to write sentences about one or more of them, so it can still
write about each alone.
"""

from __future__ import annotations

import argparse
import sys
from collections import Counter

import numpy as np
import pandas as pd


def plan(
    labels: list[str],
    requests: int,
    per_request: int,
    existing: Counter,
    per_offer: float,
    seed: int,
    uniform: bool = False,
) -> list[dict]:
    rng = np.random.default_rng(seed)
    offered = np.array(
        [existing.get(label, 0) / per_offer for label in labels], dtype=float
    )
    plans = []
    for n in range(requests):
        weights = (
            np.ones(len(labels)) if uniform else np.exp(-(offered - offered.min()))
        )
        picked = rng.choice(
            len(labels), size=per_request, replace=False, p=weights / weights.sum()
        )
        offered[picked] += 1
        plans.append(
            {"request_id": f"req-{n:05d}", "labels": sorted(labels[i] for i in picked)}
        )
    return plans


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("labels")
    parser.add_argument("out")
    parser.add_argument("--requests", type=int, default=1000)
    parser.add_argument("--per-request", type=int, default=8)
    parser.add_argument("--existing")
    parser.add_argument("--examples-per-offer", type=float, default=4.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--uniform", action="store_true")
    parser.add_argument("--background", type=int, default=0)
    args = parser.parse_args(argv)
    labels = sorted(pd.read_csv(args.labels)["label"])
    existing: Counter = Counter()
    if args.existing:
        existing.update(
            x for ls in pd.read_json(args.existing, lines=True)["labels"] for x in ls
        )
    plans = plan(
        labels,
        args.requests,
        args.per_request,
        existing,
        args.examples_per_offer,
        args.seed,
        args.uniform,
    )
    plans += [
        {"request_id": f"bg-{n:05d}", "labels": []} for n in range(args.background)
    ]
    pd.DataFrame(plans).to_json(
        args.out, orient="records", lines=True, force_ascii=False
    )
    counts = Counter(x for p in plans for x in p["labels"])
    print(f"requests       {len(plans)}")
    if args.background:
        print(f"background     {args.background}")
    if counts:
        print(f"labels offered {len(counts)} of {len(labels)}")
        print(f"least offered  {min(counts.values())}")
        print(f"most offered   {max(counts.values())}")
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
