"""Assemble training data in the realistic input formats.

Sources:
  old    the first data rounds (data/v1 ... data/v7 examples.jsonl): the search text as written, the old structured
         filters as interface facets, the teacher's original verdict and narrative
  real   realistic-search chunks (data/real/c*/examples.jsonl), narrated from the full new format
Splits follow the theme (generate.split_of), as in the first rounds, so validation and test themes are unseen.
Each row carries two inputs: "input" in the full format (Understood and Checks lines) and "input_min" in the minimal
format (no query understanding); train.py reads "input", so the minimal variant is written as its own directory.

  python -m relevance_explainer.synth.assemble_v2 --out data/v2 --old-share 0.6
"""
from __future__ import annotations

import argparse
import json
import random
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

from . import generate as G
from .realistic import item_first, render_v2, render_v2min

ROUNDS = [f"data/v{i}" for i in range(1, 8)]
_U = None


def _init():
    global _U
    from .understand import Understander
    _U = Understander()


def _render_old(args):
    r, sim = args
    p = {"query": r["query"], "language": r["language"], "record": r["record"], "facet_values": r.get("filters") or {}}
    parsed = _U.parse(r["query"], p["facet_values"])
    return render_v2(p, _U, parsed, sim), render_v2min(p, sim)


def _render_real(r: dict):
    """Re-render a realistic example with the current query understanding. Returns None when an earlier parse found a
    place the current one does not and the narrative names it (the narrative followed a wrong Understood line)."""
    from .understand import norm
    parsed = _U.parse(r["query"], r.get("facet_values"))
    new = {x.get("iso3") or x.get("code") for x in parsed["places"]}
    for x in (r.get("parsed") or {}).get("places", []):
        if (x.get("iso3") or x.get("code")) not in new and norm(x["name"]) in norm(r["narrative"]):
            return None
    return render_v2(r, _U, parsed, r["similarity"]), render_v2min(r, r["similarity"])


def load_old() -> list[dict]:
    seen, rows = set(), []
    for d in ROUNDS:
        path = Path(d) / "examples.jsonl"
        if not path.exists():
            continue
        for line in path.read_text().splitlines():
            r = json.loads(line)
            key = (r["query"], r["record"]["id"])
            if "publication year" in r["input"] or key in seen:
                continue
            seen.add(key)
            rows.append(r)
    return rows


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="data/v2")
    ap.add_argument("--real", nargs="*", default=None, help="realistic chunk directories (default: all data/real/c*)")
    ap.add_argument("--old-share", type=float, default=0.6, help="largest share of old-round rows in the training split")
    ap.add_argument("--procs", type=int, default=24)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    rng = random.Random(a.seed)
    out = Path(a.out)

    old = load_old()
    from .signals import item_text, similarities
    sims = similarities([r["query"] for r in old], [item_text(r["record"]) for r in old])
    with ProcessPoolExecutor(a.procs, initializer=_init) as ex:
        rendered = list(ex.map(_render_old, zip(old, map(float, sims)), chunksize=256))
    rows = [{"input": v2, "input_min": vm, "output": f"{r['verdict']}\n{r['narrative']}", "verdict": r["verdict"],
             "source": r["source"], "language": r["language"], "seed": r["seed"], "origin": "old"}
            for r, (v2, vm) in zip(old, rendered)]

    real_dirs = a.real if a.real is not None else sorted(str(p) for g in ("c*", "t*") for p in Path("data/real").glob(g) if (p / "examples.jsonl").exists())
    seen, real = set(), []
    for d in real_dirs:
        for line in (Path(d) / "examples.jsonl").read_text().splitlines():
            r = json.loads(line)
            key = (r["query"], r["record"]["id"])
            if key not in seen:
                seen.add(key)
                real.append(r)
    with ProcessPoolExecutor(a.procs, initializer=_init) as ex:
        rer = list(ex.map(_render_real, real, chunksize=256))
    print(f"realistic examples: {len(real)}; dropped after re-rendering: {sum(x is None for x in rer)}")
    for r, x in zip(real, rer):
        if x is None:
            continue
        r["input_v2"], r["input_v2min"] = x
        rows.append({"input": r["input_v2"], "input_min": r["input_v2min"], "output": f"{r['verdict']}\n{r['narrative']}",
                         "verdict": r["verdict"], "source": r["record"]["kind"], "language": r["language"], "seed": r["topic"],
                         "origin": "translated" if r.get("mode") == "translated" else "real", "style": r.get("style"), "writer": r.get("writer", "qwen3.8-27b"), "mode": r.get("mode")})

    splits = {"train": [], "val": [], "test": []}
    for r in rows:
        splits[G.split_of(r["seed"])].append(r)
    tr = splits["train"]
    n_real = sum(r["origin"] != "old" for r in tr)
    n_old = sum(r["origin"] == "old" for r in tr)
    keep_old = min(n_old, int(n_real * a.old_share / max(1e-9, 1 - a.old_share))) if n_real else n_old
    olds = [r for r in tr if r["origin"] == "old"]
    rng.shuffle(olds)
    splits["train"] = [r for r in tr if r["origin"] != "old"] + olds[:keep_old]
    rng.shuffle(splits["train"])
    for r in rows:
        r["input_itemfirst"] = item_first(r["input"])
    for variant, key in (("", "input"), ("_min", "input_min"), ("_itemfirst", "input_itemfirst")):
        d = Path(str(out) + variant)
        d.mkdir(parents=True, exist_ok=True)
        for name, rs in splits.items():
            with (d / f"{name}.jsonl").open("w") as f:
                for r in rs:
                    f.write(json.dumps({**{k: v for k, v in r.items() if k not in ("input", "input_min", "input_itemfirst")}, "input": r[key]}, ensure_ascii=False) + "\n")
    for name, rs in splits.items():
        print(name, len(rs), dict(Counter(r["origin"] for r in rs)), dict(Counter(r["verdict"] for r in rs)))


if __name__ == "__main__":
    main()
