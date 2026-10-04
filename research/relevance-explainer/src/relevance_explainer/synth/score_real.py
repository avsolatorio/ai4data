"""Score a run's generations on the realistic test set against the teacher, majority and unanimous labels.

  python -m relevance_explainer.synth.score_real --data data/eval_real/full --run runs/v2a-full --tag realtest
"""
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--run", required=True)
    ap.add_argument("--tag", default="realtest")
    ap.add_argument("--prec", default="int8")
    a = ap.parse_args()
    rows = [json.loads(l) for l in (Path(a.data) / "test.jsonl").read_text().splitlines()]
    gens = [json.loads(l) for l in (Path(a.run) / f"{a.tag}_generations_{a.prec}.jsonl").read_text().splitlines()]
    assert len(rows) == len(gens), (len(rows), len(gens))
    pv = [g["generated"].split("\n")[0].strip() for g in gens]
    def acc(idx, key):
        idx = [i for i in idx if key(rows[i])]
        return (sum(pv[i] == key(rows[i]) for i in idx) / len(idx), len(idx)) if idx else (float("nan"), 0)
    allidx = range(len(rows))
    res = {"teacher": acc(allidx, lambda r: r["verdict"]), "majority": acc(allidx, lambda r: r.get("majority")),
           "unanimous": acc([i for i in allidx if rows[i].get("unanimous")], lambda r: r["verdict"])}
    for name, key in (("style", "style"), ("relation", "relation"), ("mode", "mode"), ("language", "language"), ("source", "source")):
        g = defaultdict(list)
        for i, r in enumerate(rows):
            g[r.get(key) or "-"].append(i)
        res[f"by_{name}"] = {k: round(acc(v, lambda r: r["verdict"])[0], 3) for k, v in sorted(g.items())}
    (Path(a.run) / f"{a.tag}_score.json").write_text(json.dumps(res, indent=1))
    print(f"{a.run}: teacher {res['teacher'][0]:.3f} (n={res['teacher'][1]}) | majority {res['majority'][0]:.3f} | unanimous {res['unanimous'][0]:.3f} (n={res['unanimous'][1]})")
    for k in ("by_style", "by_mode", "by_relation"):
        print(f"  {k}: {res[k]}")


if __name__ == "__main__":
    main()
