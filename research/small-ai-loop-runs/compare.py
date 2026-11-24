"""Paired bootstrap over the report set: difference between two run folders in the score averaged over seeds.

The score is micro F1 for a multi-label task and accuracy for a single-label task.
python compare.py RUN_A RUN_B [TAG_A TAG_B]   prints (A minus B, lower 2.5%, upper 97.5%, share of resamples above 0)"""
import json, sys
from pathlib import Path
import numpy as np, pandas as pd
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
from ai4data.small_ai import load_task  # noqa: E402
from ai4data.small_ai.evaluate import choose  # noqa: E402

def matrices(run, tag="final"):
    run = Path(run); task = load_task(json.loads((run / "run.json").read_text())["task"])
    gold = pd.read_json(run / "reference/report.jsonl", lines=True, dtype={"id": str})
    fin = json.loads((run / f"{tag}.json").read_text())
    measured = sorted({x for ls in gold["labels"] for x in ls}) if task.multi_label else None
    out = []
    for seed, v in fin["seeds"].items():
        pred = pd.read_json(run / tag / f"seed{seed}" / "report_scores.jsonl", lines=True, dtype={"id": str})
        d = gold.merge(pred, on="id")
        if task.multi_label:
            t = np.array([[x in ls for x in measured] for ls in d["labels"]])
            g = np.array([[x in set(choose(s, v["threshold"], 5)) for x in measured] for s in d["scores"]])
            out.append((t, g))
        else:
            c = np.array([ls[0] == max(s, key=s.get) for ls, s in zip(d["labels"], d["scores"])])
            out.append((c, None))
    return task.multi_label, out

def stat(multi, mats, idx):
    vals = []
    for t, g in mats:
        if multi:
            tt, gg = t[idx], g[idx]; tp = (tt & gg).sum()
            vals.append(2 * tp / (tt.sum() + gg.sum()))
        else:
            vals.append(t[idx].mean())
    return np.mean(vals)

def compare(a, b, tag_a="final", tag_b="final", n=2000):
    multi, ma = matrices(a, tag_a); _, mb = matrices(b, tag_b)
    N = len(ma[0][0]); rng = np.random.default_rng(0); all_ = np.arange(N)
    diffs = []
    for _ in range(n):
        idx = rng.integers(0, N, N)
        diffs.append(stat(multi, ma, idx) - stat(multi, mb, idx))
    d0 = stat(multi, ma, all_) - stat(multi, mb, all_)
    lo, hi = np.percentile(diffs, [2.5, 97.5])
    return round(d0, 4), round(lo, 4), round(hi, 4), round(float(np.mean(np.array(diffs) > 0)), 3)

if __name__ == "__main__":
    print(compare(*sys.argv[1:3], *(sys.argv[3:5] if len(sys.argv) > 3 else [])))
