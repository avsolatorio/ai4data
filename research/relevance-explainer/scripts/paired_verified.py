"""Paired comparison of two format-C runs' decoding tests (scripts/fmt_decode_test.py) on the same rows: share of
answers with every mention and reference verified, for one column, with McNemar's exact test.

  python scripts/paired_verified.py runs/final-C runs/final-C-dpo [--col greedy]
"""
import argparse
import json
from math import comb
from pathlib import Path

from relevance_explainer.claims import from_model, verify

ap = argparse.ArgumentParser()
ap.add_argument("a"); ap.add_argument("b"); ap.add_argument("--col", default="greedy")
ap.add_argument("--file", default="decode_test_itemfirst_nosim_v4.jsonl")
a = ap.parse_args()


def ok(run):
    return {d["input"]: verify(from_model(d[a.col]), d["input"]).all_verified
            for d in map(json.loads, open(Path(run) / a.file))}


va, vb = ok(a.a), ok(a.b)
ks = sorted(va.keys() & vb.keys())
oa = sum(va[k] and not vb[k] for k in ks)
ob = sum(vb[k] and not va[k] for k in ks)
n = oa + ob
p = min(1.0, 2 * sum(comb(n, i) for i in range(min(oa, ob) + 1)) / 2 ** n) if n else 1.0
print(f"{len(ks)} rows, {a.col}: all verified {Path(a.a).name} {100 * sum(va[k] for k in ks) / len(ks):.1f}%, "
      f"{Path(a.b).name} {100 * sum(vb[k] for k in ks) / len(ks):.1f}%; only {Path(a.a).name} {oa}, "
      f"only {Path(a.b).name} {ob}, McNemar p = {p:.4f}")
