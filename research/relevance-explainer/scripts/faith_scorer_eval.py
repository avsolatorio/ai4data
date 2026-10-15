"""Evaluate a fitted faithfulness scorer (<scorer run>/faith_scorer.json) on a run's judged realistic-test answers:
the greedy answers of the decoding test (scripts/judge_prep_decode.py) with the teacher judge's scores
(tiny/judge.py, <run>/judge_<tag>_int8.json, same seed-0 sample of rows). These are unselected answers on inputs
outside training, unlike the fitting labels. Reports AUROC with a bootstrap interval, Brier score, ECE and the judge's
rate above and below thresholds.

  CUDA_VISIBLE_DEVICES=1 python scripts/faith_scorer_eval.py runs/final-C [--scorer runs/final-C] [--tag realnb] [--suffix _gemma4]
"""
import argparse
import json
import random
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import brier_score_loss, roc_auc_score
from tokenizers import Tokenizer

from relevance_explainer.distill.onpolicy import EOS, SEP
from relevance_explainer.faith import FaithScorer, model_features, verifier_features
from relevance_explainer.tiny.model import Config, TinyLM

ap = argparse.ArgumentParser()
ap.add_argument("run"); ap.add_argument("--scorer", default=None); ap.add_argument("--tag", default="realnb")
ap.add_argument("--suffix", default="", help="judge file suffix, e.g. _gemma4 for the Gemma 4 judge")
a = ap.parse_args()
run = Path(a.run)
rows = [json.loads(l) for l in (run / f"{a.tag}_generations_int8.jsonl").read_text().splitlines()]
judge = json.loads((run / f"judge_{a.tag}_int8{a.suffix}.json").read_text())
n = max(x["k"] for x in judge["items"]) + 1
rows = random.Random(0).sample(rows, n)  # the judge's sample (tiny/judge.py)
lab = {x["k"]: x["score"]["faithful"] for x in judge["items"] if x["who"] == "student" and x["score"]}
ks = sorted(lab)
ans = [rows[k]["raw"].strip() for k in ks]

ck = torch.load(run / "model_fp32.pt", map_location="cpu")
model = TinyLM(Config(**ck["config"]))
model.load_state_dict(ck["state"])
model = model.cuda().eval()
tk = Tokenizer.from_file(str(run / "tokenizer.json"))
verdict = [x.split("\n", 1)[0].strip() for x in ans]
prompts = [tk.encode(rows[k]["input"] + SEP + v + "\n").ids for k, v in zip(ks, verdict)]
answers = [tk.encode(x.split("\n", 1)[1] if "\n" in x else "").ids + [EOS] for x in ans]
gen, hid = model_features(model, prompts, answers)
ver = np.array([verifier_features(x, rows[k]["input"]) for k, x in zip(ks, ans)])
p = FaithScorer(Path(a.scorer or run) / "faith_scorer.json").proba(ver, gen, hid)
y = np.array([lab[k] >= 4 for k in ks], int)
ok = ~np.isnan(p)
p, y = p[ok], y[ok]
rng = np.random.default_rng(0)
boot = [roc_auc_score(y[s], p[s]) for s in (rng.integers(0, len(y), len(y)) for _ in range(2000)) if y[s].min() < y[s].max()]
e = np.minimum((p * 10).astype(int), 9)
ece = sum(abs(p[e == b].mean() - y[e == b].mean()) * (e == b).mean() for b in range(10) if (e == b).any())
print(f"{len(y)} answers, judge 4-5 rate {y.mean():.3f}, mean score {p.mean():.3f}")
print(f"AUROC {roc_auc_score(y, p):.3f} (95% {np.percentile(boot, 2.5):.3f} to {np.percentile(boot, 97.5):.3f}), "
      f"Brier {brier_score_loss(y, p):.3f}, ECE {ece:.3f}")
for thr in (0.5, 0.7, 0.8, 0.9):
    sel = p >= thr
    print(f"p >= {thr}: {sel.mean():.3f} of answers, judge 4-5 among them {y[sel].mean():.3f}; "
          f"below: {y[~sel].mean() if (~sel).any() else float('nan'):.3f} (n={int((~sel).sum())})")
