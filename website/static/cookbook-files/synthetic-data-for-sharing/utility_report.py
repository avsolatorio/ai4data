"""Compare a synthetic table with the real one on marginals, associations,
and a target analysis.

Reads the real and the synthetic CSV (same columns) and reports:

    marginals      for each categorical column the total variation
                   distance between the two distributions; for each
                   numeric column the difference of the medians and of
                   the interquartile ranges relative to the real ones
    associations   for each pair of categorical columns the difference
                   in Cramér's V; for numeric pairs the difference in
                   Pearson correlation
    target         a statistic users compute (here the employment rate
                   by sex and the median income by education) in both
                   files side by side

Columns with at most 12 distinct values are treated as categorical.
Empty values are their own category for categorical columns and are
dropped for numeric ones. Standard library only.

Usage:
    python utility_report.py real_sample.csv synthetic_sample.csv

What this does not do: these are general-purpose measures. The utility
that matters is whether the analyses users will run give the same
answers, which the target section shows for two statistics; the
organization adds the analyses its users run. A synthetic file that
scores well on marginals and badly on associations is the usual
failure of independent synthesis, which the running example shows.
"""

from __future__ import annotations

import argparse
import csv
import math
import statistics
import sys
from collections import Counter
from itertools import combinations
from pathlib import Path

CATEGORICAL_MAX = 12


def load(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


def kinds(rows: list[dict[str, str]]) -> dict[str, str]:
    out = {}
    for c in rows[0]:
        vals = {r[c] for r in rows}
        numeric = all(
            v == "" or v.replace(".", "", 1).lstrip("-").isdigit() for v in vals
        )
        out[c] = (
            "categorical" if len(vals) <= CATEGORICAL_MAX or not numeric else "numeric"
        )
    return out


def tvd(a: list[str], b: list[str]) -> float:
    ca, cb = Counter(a), Counter(b)
    keys = set(ca) | set(cb)
    return 0.5 * sum(abs(ca[k] / len(a) - cb[k] / len(b)) for k in keys)


def cramers_v(x: list[str], y: list[str]) -> float:
    n = len(x)
    cx, cy, cxy = Counter(x), Counter(y), Counter(zip(x, y, strict=True))
    chi2 = 0.0
    for (a, b), o in cxy.items():
        e = cx[a] * cy[b] / n
        chi2 += (o - e) ** 2 / e
    for a in cx:
        for b in cy:
            if (a, b) not in cxy:
                chi2 += cx[a] * cy[b] / n
    k = min(len(cx), len(cy)) - 1
    return math.sqrt(chi2 / (n * k)) if k > 0 else 0.0


def pearson(x: list[float], y: list[float]) -> float:
    if len(x) < 3:
        return 0.0
    mx, my = statistics.fmean(x), statistics.fmean(y)
    sx = math.sqrt(sum((v - mx) ** 2 for v in x))
    sy = math.sqrt(sum((v - my) ** 2 for v in y))
    return (
        sum((a - mx) * (b - my) for a, b in zip(x, y, strict=True)) / (sx * sy)
        if sx and sy
        else 0.0
    )


def numeric(rows: list[dict[str, str]], c: str) -> list[float]:
    return [float(r[c]) for r in rows if r[c] != ""]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("real", type=Path)
    parser.add_argument("synthetic", type=Path)
    args = parser.parse_args(argv)
    real, synth = load(args.real), load(args.synthetic)
    kind = kinds(real)
    print(f"real {len(real)} rows, synthetic {len(synth)} rows, {len(kind)} columns\n")

    print("marginals")
    for c, k in kind.items():
        if k == "categorical":
            print(
                f"  {c:<12} categorical  TVD {tvd([r[c] for r in real], [r[c] for r in synth]):.3f}"
            )
        else:
            a, b = numeric(real, c), numeric(synth, c)
            qa = statistics.quantiles(a, n=4)
            qb = statistics.quantiles(b, n=4)
            med = (qb[1] - qa[1]) / qa[1] if qa[1] else 0.0
            iqr_a, iqr_b = qa[2] - qa[0], qb[2] - qb[0]
            print(
                f"  {c:<12} numeric      median diff {med:+.1%}, IQR diff {(iqr_b - iqr_a) / iqr_a if iqr_a else 0:+.1%}"
            )

    print("\nassociations (real vs synthetic)")
    cats = [c for c, k in kind.items() if k == "categorical"]
    nums = [c for c, k in kind.items() if k == "numeric"]
    for a, b in combinations(cats, 2):
        va = cramers_v([r[a] for r in real], [r[b] for r in real])
        vb = cramers_v([r[a] for r in synth], [r[b] for r in synth])
        flag = "  <- association lost" if va - vb > 0.1 else ""
        print(f"  {a} x {b:<12} Cramer's V {va:.2f} vs {vb:.2f}{flag}")
    for a, b in combinations(nums, 2):
        pa = [(float(r[a]), float(r[b])) for r in real if r[a] != "" and r[b] != ""]
        pb = [(float(r[a]), float(r[b])) for r in synth if r[a] != "" and r[b] != ""]
        ra = pearson([x for x, _ in pa], [y for _, y in pa])
        rb = pearson([x for x, _ in pb], [y for _, y in pb])
        print(f"  {a} x {b:<12} correlation {ra:+.2f} vs {rb:+.2f}")

    print("\ntarget analyses")
    for sex in ("1", "2"):
        for name, rows in (("real", real), ("synthetic", synth)):
            group = [
                r
                for r in rows
                if r["sex"] == sex and r["age"] != "" and 15 <= float(r["age"]) <= 64
            ]
            rate = (
                sum(1 for r in group if r["lfs_status"] == "1") / len(group)
                if group
                else 0.0
            )
            print(
                f"  employment rate, sex {sex}, ages 15-64: {name:<9} {rate:.3f} (n={len(group)})"
            )
    for educ in ("0", "1", "2", "3"):
        vals = {
            name: numeric([r for r in rows if r["educ"] == educ], "income")
            for name, rows in (("real", real), ("synthetic", synth))
        }
        print(
            f"  median income, education {educ}: real {statistics.median(vals['real']) if vals['real'] else 0:.0f}, synthetic {statistics.median(vals['synthetic']) if vals['synthetic'] else 0:.0f}"
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
