"""Live faithfulness scorer for format C: the probability that the teacher judge would rate an answer's explanation
faithful 4 or 5, from signals available at inference without a teacher.

Labels: the judged samples of the DPO round (distill/onpolicy_c.py score: per input the 2 best and the worst of 8
samples by the verifier, so the label set over-represents both ends). Feature sets, each a logistic regression:
  verifier   all verified, verified share, unverified mentions, reference errors, references, explanation words
  generator  the generating model's log-probabilities of its own answer tokens: mean, minimum, 5th percentile, share
             below log(0.1), mean entropy, answer tokens
  hidden     the generator's final hidden state (after the last norm), mean over the answer tokens and at the last token
  combined   all three
The split is by input (80/20), so no input appears on both sides. Reports AUROC, Brier score and ECE on the held-out
inputs, with a bootstrap interval on the AUROC (resampled by input), and writes the combined model to the run.

  CUDA_VISIBLE_DEVICES=1 python scripts/faith_scorer.py runs/final-C data/final/C onpolicy_c_final-C_scored.jsonl
"""
import argparse
import json
from pathlib import Path

import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from tokenizers import Tokenizer

from relevance_explainer.claims import plain_output
from relevance_explainer.distill.onpolicy import EOS, SEP, _read
from relevance_explainer.faith import model_features
from relevance_explainer.tiny.model import Config, TinyLM

ap = argparse.ArgumentParser()
ap.add_argument("run"); ap.add_argument("data"); ap.add_argument("scored")
ap.add_argument("--batch", type=int, default=48); ap.add_argument("--seed", type=int, default=0)
ap.add_argument("--out", default=None, help="output file (default <run>/faith_scorer.json)")
ap.add_argument("--cache", default=None, help="feature cache (default <run>/faith_features_<scored>.npz)")
a = ap.parse_args()
run, data = Path(a.run), Path(a.data)
cache = Path(a.cache) if a.cache else run / f"faith_features_{Path(a.scored).stem}.npz"

rows = _read(data / "train.jsonl")
items = []  # (input index, prompt text, answer text, check, faithful)
for d in _read(data / a.scored):
    for k, s in d["scores"].items():
        if s:
            items.append((d["i"], rows[d["i"]]["input"] + SEP + d["verdict"] + "\n", d["samples"][int(k)],
                          d["checks"][int(k)], s["faithful"]))
print(f"{len(items)} judged samples from {len({x[0] for x in items})} inputs", flush=True)

if cache.exists():
    z = np.load(cache)
    gen, hid = z["gen"], z["hid"]
else:
    ck = torch.load(run / "model_fp32.pt", map_location="cpu")
    model = TinyLM(Config(**ck["config"]))
    model.load_state_dict(ck["state"])
    model = model.cuda().eval()
    tk = Tokenizer.from_file(str(run / "tokenizer.json"))
    gen, hid = model_features(model, [tk.encode(p).ids for _, p, _, _, _ in items],
                              [tk.encode(s).ids + [EOS] for _, _, s, _, _ in items], batch=a.batch)
    np.savez_compressed(cache, gen=gen, hid=hid)

ver = np.array([[c["verified"], c["share"], c["unverified"], c["errors"], c["refs"],
                 len(plain_output(s).split())] for _, _, s, c, _ in items], float)
y = np.array([f >= 4 for *_, f in items], int)
groups = np.array([x[0] for x in items])
ok = ~np.isnan(gen).any(1)
uniq = np.unique(groups)
test_in = set(np.random.default_rng(a.seed).choice(uniq, len(uniq) // 5, replace=False).tolist())
te = np.array([g in test_in for g in groups]) & ok
tr = ~np.array([g in test_in for g in groups]) & ok
print(f"label faithful>=4: {y[ok].mean():.3f} of {ok.sum()}; train {tr.sum()}, test {te.sum()}", flush=True)


def ece(p, t, bins=10):
    e = np.minimum((p * bins).astype(int), bins - 1)
    return sum(abs(p[e == b].mean() - t[e == b].mean()) * (e == b).mean() for b in range(bins) if (e == b).any())


def auc_ci(p, t, g):
    rng = np.random.default_rng(1)
    ug = np.unique(g)
    idx = {u: np.where(g == u)[0] for u in ug}
    out = []
    for _ in range(1000):
        s = np.concatenate([idx[u] for u in rng.choice(ug, len(ug))])
        if t[s].min() != t[s].max():
            out.append(roc_auc_score(t[s], p[s]))
    return np.percentile(out, [2.5, 97.5])


sets = {"verifier": ver, "generator": gen, "hidden": hid,
        "verifier+generator": np.hstack([ver, gen]), "combined": np.hstack([ver, gen, hid])}
res, models = {}, {}
for name, X in sets.items():
    m = make_pipeline(StandardScaler(), LogisticRegression(C=0.05 if X.shape[1] > 50 else 1.0, max_iter=3000))
    m.fit(X[tr], y[tr])
    p = m.predict_proba(X[te])[:, 1]
    lo, hi = auc_ci(p, y[te], groups[te])
    res[name] = {"auroc": roc_auc_score(y[te], p), "auroc_ci": [lo, hi], "brier": brier_score_loss(y[te], p),
                 "ece": ece(p, y[te]), "features": X.shape[1]}
    models[name] = (m, p)
    print(f"| {name} | {X.shape[1]} | {res[name]['auroc']:.3f} ({lo:.3f} to {hi:.3f}) | {res[name]['brier']:.3f} "
          f"| {res[name]['ece']:.3f} |", flush=True)
# precision at the scorer's top bands: share of answers above a threshold and how many of those the judge rated 4-5
p = models["combined"][1]
for thr in (0.5, 0.7, 0.8, 0.9):
    sel = p >= thr
    print(f"combined p >= {thr}: {sel.mean():.3f} of answers, judge 4-5 among them {y[te][sel].mean():.3f}; "
          f"below: {y[te][~sel].mean() if (~sel).any() else float('nan'):.3f}", flush=True)
m = models["combined"][0]
sc, lr = m.named_steps["standardscaler"], m.named_steps["logisticregression"]
Path(a.out or run / "faith_scorer.json").write_text(json.dumps({
    "label": "teacher judge faithful >= 4", "scored": a.scored, "results": res,
    "features": ["verified", "share", "unverified", "errors", "refs", "words",
                 "lp_mean", "lp_min", "lp_p05", "lp_share_below_0.1", "entropy_mean", "answer_tokens",
                 f"hidden_mean[{hid.shape[1] // 2}]", f"hidden_last[{hid.shape[1] // 2}]"],
    "mean": sc.mean_.tolist(), "scale": sc.scale_.tolist(), "coef": lr.coef_[0].tolist(),
    "intercept": float(lr.intercept_[0])}, indent=1))
print(f"wrote {a.out or run / 'faith_scorer.json'}")
