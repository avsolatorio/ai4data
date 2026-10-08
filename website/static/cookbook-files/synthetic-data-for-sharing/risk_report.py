"""Measure the disclosure risk of a synthetic table against the real one.

Reads the real and the synthetic CSV (same columns) and reports three
checks that a release of synthetic microdata needs:

    exact copies      synthetic records identical to a real record on
                      every column (a synthesizer that memorized)
    closest record    for each synthetic record the normalized distance
                      to its closest real record, compared with the same
                      distance computed between two halves of the real
                      data; synthetic records much closer to real ones
                      than real records are to each other indicate
                      copying
    attribute inference  how well a sensitive attribute (income band) of
                      real records can be predicted from the
                      quasi-identifiers through the synthetic data (a
                      nearest-neighbour lookup), compared with a baseline
                      that always predicts the most common band

Standard library only. Quasi-identifiers and the sensitive attribute are
arguments; the defaults match the running example.

Usage:
    python risk_report.py real_sample.csv synthetic_sample.csv \
        --quasi region sex age educ --sensitive income

What this does not do: three checks on three hundred records are a
demonstration of the kinds of evidence a release needs. The statistical
disclosure control unit sets the thresholds and may require membership
inference tests and a formal privacy accounting, which the risk chapter
points to. Removing exact copies is the minimum; a synthesizer that
produced them needs a parameter change and a rerun.
"""

from __future__ import annotations

import argparse
import csv
import random
import statistics
import sys
from collections import Counter
from pathlib import Path


def load(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


def band(value: str) -> str:
    if value == "":
        return "none"
    v = float(value)
    return "low" if v < 500 else "mid" if v < 1200 else "high"


def distance(
    a: dict[str, str], b: dict[str, str], quasi: list[str], ranges: dict[str, float]
) -> float:
    d = 0.0
    for q in quasi:
        if q in ranges:
            d += (
                abs(float(a[q]) - float(b[q])) / ranges[q]
                if a[q] != "" and b[q] != ""
                else 1.0
            )
        else:
            d += 0.0 if a[q] == b[q] else 1.0
    return d / len(quasi)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("real", type=Path)
    parser.add_argument("synthetic", type=Path)
    parser.add_argument("--quasi", nargs="+", default=["region", "sex", "age", "educ"])
    parser.add_argument("--sensitive", default="income")
    parser.add_argument("--seed", type=int, default=7)
    args = parser.parse_args(argv)
    real, synth = load(args.real), load(args.synthetic)
    rng = random.Random(args.seed)

    keys = {tuple(r.values()) for r in real}
    copies = [i for i, s in enumerate(synth) if tuple(s.values()) in keys]
    print(f"real {len(real)} rows, synthetic {len(synth)} rows")
    print(
        f"\nexact copies: {len(copies)} synthetic record(s) identical to a real record"
        + (f" (rows {', '.join(str(i + 1) for i in copies)})" if copies else "")
    )

    ranges = {}
    for q in args.quasi:
        vals = [
            float(r[q])
            for r in real
            if r[q] != "" and r[q].replace(".", "", 1).isdigit()
        ]
        if len(vals) == len(real) and len(set(vals)) > 12:
            ranges[q] = (max(vals) - min(vals)) or 1.0
    half = rng.sample(range(len(real)), len(real) // 2)
    a, b = (
        [real[i] for i in half],
        [real[i] for i in range(len(real)) if i not in set(half)],
    )
    d_syn = [min(distance(s, r, args.quasi, ranges) for r in real) for s in synth]
    d_real = [min(distance(x, r, args.quasi, ranges) for r in b) for x in a]
    print("\nclosest record distance on the quasi-identifiers (0 = identical)")
    print(
        f"  synthetic to real:      median {statistics.median(d_syn):.3f}, share at 0: {sum(1 for d in d_syn if d == 0) / len(d_syn):.2f}"
    )
    print(
        f"  real half to other half: median {statistics.median(d_real):.3f}, share at 0: {sum(1 for d in d_real if d == 0) / len(d_real):.2f}"
    )
    print(
        "  a synthetic file much closer to the real one than the real halves are to each other has copied records"
    )

    bands_real = Counter(band(r[args.sensitive]) for r in real)
    majority = bands_real.most_common(1)[0][0]
    hits = 0
    for r in real:
        nearest = min(synth, key=lambda s: distance(r, s, args.quasi, ranges))
        hits += band(nearest[args.sensitive]) == band(r[args.sensitive])
    print(
        f"\nattribute inference of {args.sensitive} band from the quasi-identifiers through the synthetic file"
    )
    print(
        f"  nearest-neighbour accuracy {hits / len(real):.2f} vs majority baseline {bands_real[majority] / len(real):.2f}"
    )
    print(
        "  an accuracy far above the baseline means the synthetic file reveals the attribute for people like the quasi-identifiers; the disclosure control unit sets the limit"
    )
    return 1 if copies else 0


if __name__ == "__main__":
    sys.exit(main())
