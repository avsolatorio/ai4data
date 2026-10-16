"""Calibration file for serving a tagged run: the verdict temperature and the split-conformal thresholds, fitted on
rows of the training-distribution validation set and checked on the realistic test (CPU, int8, as in the app).
The realistic test's val.jsonl is a copy of its test.jsonl, so it cannot be the calibration set.

Writes <run>/calibration.json: {"temperature", "conformal": {"0.90": q, "0.95": q}, "fitted_on", "check"}. A verdict v
is in the set when 1 - P_T(v) <= q, with P_T the probabilities at temperature T (scripts/uncertainty_check.py).

  python scripts/calibrate_c.py runs/final-C-dpo data/final/C/val.jsonl data/eval_real/itemfirst_nosim_v4/test.jsonl [--n 3000]
"""
import argparse
import json
import math
import random
from pathlib import Path

import numpy as np
import torch
from tokenizers import Tokenizer

from relevance_explainer.claims import sanitize_input
from relevance_explainer.tiny.model import Config, TinyLM
from relevance_explainer.tiny.train import SEP, quantize

ap = argparse.ArgumentParser()
ap.add_argument("run"); ap.add_argument("cal"); ap.add_argument("test")
ap.add_argument("--n", type=int, default=3000); ap.add_argument("--threads", type=int, default=8)
a = ap.parse_args()
torch.set_num_threads(a.threads)
V = ["relevant", "partly_relevant", "irrelevant"]
run = Path(a.run)
ck = torch.load(run / "model_fp32.pt", map_location="cpu")
model = TinyLM(Config(**ck["config"]))
model.load_state_dict(ck["state"])
model, _ = quantize(model.eval(), 8)
tk = Tokenizer.from_file(str(run / "tokenizer.json"))
first = [tk.encode(v).ids[0] for v in V]


def logits(rows):
    z, y = [], []
    with torch.no_grad():
        for r in rows:
            p = tk.encode(sanitize_input(r["input"]) + SEP).ids[-model.c.max_len:]
            prefix = torch.tensor([len(p)]) if model.c.prefix_lm else None
            z.append(model(torch.tensor([p]), prefix=prefix)[0, -1].float()[first].numpy())
            y.append(V.index(r.get("verdict") or r["output"].split("\n")[0].strip()))
    return np.array(z), np.array(y)


def softmax(z, t=1.0):
    z = z / t
    e = np.exp(z - z.max(1, keepdims=True))
    return e / e.sum(1, keepdims=True)


def nll(p, y):
    return float(-np.log(np.clip(p[np.arange(len(y)), y], 1e-12, 1)).mean())


def ece(p, y, bins=10):
    conf, pred = p.max(1), p.argmax(1)
    return sum(((conf > b / bins) & (conf <= (b + 1) / bins)).mean()
               * abs((pred[m] == y[m]).mean() - conf[m].mean())
               for b in range(bins) if (m := (conf > b / bins) & (conf <= (b + 1) / bins)).any())


cal_rows = [json.loads(l) for l in open(a.cal)]
cal_rows = random.Random(0).sample(cal_rows, min(a.n, len(cal_rows)))
zc, yc = logits(cal_rows)
zt, yt = logits([json.loads(l) for l in open(a.test)])
ts = np.exp(np.linspace(math.log(0.3), math.log(5), 200))
t = float(ts[np.argmin([nll(softmax(zc, x), yc) for x in ts])])
pc, pt = softmax(zc, t), softmax(zt, t)
out = {"temperature": round(t, 4), "conformal": {}, "fitted_on": f"{a.cal} ({len(yc)} rows, seed 0)", "check": {"data": a.test}}
chk = out["check"]
chk["raw"] = {"accuracy": float((softmax(zt).argmax(1) == yt).mean()), "ece": float(ece(softmax(zt), yt)), "nll": nll(softmax(zt), yt)}
chk["scaled"] = {"accuracy": float((pt.argmax(1) == yt).mean()), "ece": float(ece(pt, yt)), "nll": nll(pt, yt)}
for level in (0.90, 0.95):
    s = 1 - pc[np.arange(len(yc)), yc]
    q = float(np.quantile(s, min(1.0, math.ceil((len(yc) + 1) * level) / len(yc)), method="higher"))
    sets = (1 - pt) <= q
    one = sets.sum(1) == 1
    out["conformal"][f"{level:.2f}"] = round(q, 6)
    chk[f"{level:.2f}"] = {"coverage": float(sets[np.arange(len(yt)), yt].mean()), "mean_size": float(sets.sum(1).mean()),
                           "single": float(one.mean()), "single_accuracy": float((pt.argmax(1)[one] == yt[one]).mean())}
(run / "calibration.json").write_text(json.dumps(out, indent=1))
print(json.dumps(out, indent=1))
