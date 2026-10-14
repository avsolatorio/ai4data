"""Paired comparison of two runs' verdicts on the same test set: accuracy of each, rows where only one is right, and
an exact two-sided McNemar p-value.

  python scripts/paired_verdicts.py runs/v2g-from-dapt-4ep runs/fmt-A-full --set realtest_v4
"""
import argparse
import json
from math import comb
from pathlib import Path

ap = argparse.ArgumentParser()
ap.add_argument("a"); ap.add_argument("b")
ap.add_argument("--set", default="realtest_v4")
a = ap.parse_args()


def load(run: str) -> dict:
    out = {}
    for line in open(Path(run) / f"{a.set}_generations_int8.jsonl"):
        g = json.loads(line)
        out[g["input"].replace("< |", "<|")] = (g["generated"].split("\n")[0].strip(), g["reference"].split("\n")[0].strip())
    return out


ra, rb = load(a.a), load(a.b)
keys = ra.keys() & rb.keys()
only_a = sum(ra[k][0] == ra[k][1] and rb[k][0] != rb[k][1] for k in keys)
only_b = sum(ra[k][0] != ra[k][1] and rb[k][0] == rb[k][1] for k in keys)
n = only_a + only_b
p = min(1.0, 2 * sum(comb(n, i) for i in range(min(only_a, only_b) + 1)) / 2 ** n) if n else 1.0
acc = lambda r: sum(r[k][0] == r[k][1] for k in keys) / len(keys)  # noqa: E731
print(f"{a.set}: {len(keys)} common rows; {Path(a.a).name} {100 * acc(ra):.1f}%, {Path(a.b).name} {100 * acc(rb):.1f}%; "
      f"only {Path(a.a).name} right {only_a}, only {Path(a.b).name} right {only_b}; McNemar exact p = {p:.4f}")
