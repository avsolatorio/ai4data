"""Paired comparison of two runs' judge scores on the same examples (tiny/judge.py items, same seed and n): mean
faithfulness difference with a bootstrap 95% interval, and McNemar on "faithful 4 or 5".

  python scripts/judge_paired.py runs/fmt-s-B runs/fmt-s-C --tag realnb
"""
import argparse
import json
from math import comb
from pathlib import Path

import numpy as np

ap = argparse.ArgumentParser()
ap.add_argument("a"); ap.add_argument("b")
ap.add_argument("--tag", default="realnb")
a = ap.parse_args()


def items(run):
    d = json.loads((Path(run) / f"judge_{a.tag}_int8.json").read_text())
    return {x["k"]: x["score"] for x in d["items"] if x["who"] == "student" and x["score"]}


ia, ib = items(a.a), items(a.b)
ks = sorted(ia.keys() & ib.keys())
fa, fb = np.array([ia[k]["faithful"] for k in ks], float), np.array([ib[k]["faithful"] for k in ks], float)
rng = np.random.default_rng(0)
boot = [(fb[s] - fa[s]).mean() for s in (rng.integers(0, len(ks), len(ks)) for _ in range(5000))]
ga, gb = fa >= 4, fb >= 4
oa, ob = int((ga & ~gb).sum()), int((gb & ~ga).sum())
n = oa + ob
p = min(1.0, 2 * sum(comb(n, i) for i in range(min(oa, ob) + 1)) / 2 ** n) if n else 1.0
print(f"{len(ks)} common examples: faithful {Path(a.a).name} {fa.mean():.3f}, {Path(a.b).name} {fb.mean():.3f}, "
      f"difference {(fb - fa).mean():+.3f} (95% {np.percentile(boot, 2.5):+.3f} to {np.percentile(boot, 97.5):+.3f}); "
      f"faithful>=4 only {Path(a.a).name} {oa}, only {Path(a.b).name} {ob}, McNemar p = {p:.3f}")
