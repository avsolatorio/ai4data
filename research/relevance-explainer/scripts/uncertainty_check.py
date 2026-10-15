"""Calibration, conformal verdict sets and the expected grade for a tagged run on the realistic test (CPU, int8).

For every row: the verdict probabilities at the first output step (as in the app), and for each verdict v the grade
distribution at the step after "v\n" (the 11 <|g:N|> tokens, renormalized), giving E[g | v] and the expected grade
E[g] = sum_v P(v) E[g | v]. Labels are the teacher verdicts.
  1. Calibration of the top verdict: accuracy, mean confidence, expected calibration error (10 bins) and NLL, raw and
     after temperature scaling fitted on the calibration half.
  2. Split conformal verdict sets (score 1 - P(true verdict), temperature-scaled probabilities): coverage, mean set
     size and the share of single-verdict sets on the test half, at 90% and 95%, averaged over 50 random halvings.
  3. Ranking: pairwise concordance with the verdict as an ordinal label for the greedy grade, E[g | top verdict],
     E[g], and the expected relevance from the verdict probabilities.

  python scripts/uncertainty_check.py runs/fmt-s-C data/eval_real/itemfirst_nosim_v4/test.jsonl
"""
import json
import math
import sys
from pathlib import Path

import numpy as np
import torch
from tokenizers import Tokenizer

from relevance_explainer.claims import from_model, parse, sanitize_input
from relevance_explainer.tiny.model import Config, TinyLM
from relevance_explainer.tiny.train import SEP, quantize

torch.set_num_threads(4)
V = ["relevant", "partly_relevant", "irrelevant"]
RANK = {"irrelevant": 0, "partly_relevant": 1, "relevant": 2}
run, data = Path(sys.argv[1]), sys.argv[2]
ck = torch.load(run / "model_fp32.pt", map_location="cpu")
model = TinyLM(Config(**ck["config"]))
model.load_state_dict(ck["state"])
model, _ = quantize(model.eval(), 8)
tk = Tokenizer.from_file(str(run / "tokenizer.json"))
first = [tk.encode(v).ids[0] for v in V]
gtok = [tk.token_to_id(f"<|g:{i}|>") for i in range(11)]
rows = [json.loads(line) for line in open(data)]
dec = {json.loads(line)["input"]: json.loads(line) for line in open(run / f"decode_test_{Path(data).parent.name}.jsonl")}

# the verdict line is tokenized as at training time (the output encoded on its own)
sample_out = json.loads(open(Path("data/fmt_s/C/train.jsonl")).readline())["output"]
v0 = sample_out.split("\n")[0]
assert tk.encode(sample_out).ids[: len(tk.encode(v0 + "\n").ids)] == tk.encode(v0 + "\n").ids


def last_logits(ids: list[int]) -> torch.Tensor:
    x = torch.tensor([ids])
    prefix = torch.tensor([len(ids)]) if model.c.prefix_lm else None
    return model(x, prefix=prefix)[0, -1].float()


logit3, eg_v, greedy_grade, gold = [], [], [], []
with torch.no_grad():
    for r in rows:
        p = tk.encode(sanitize_input(r["input"]) + SEP).ids
        logit3.append(last_logits(p)[first].numpy())
        e = []
        for v in V:
            lg = last_logits(p + tk.encode(v + "\n").ids)[gtok]
            pg = torch.softmax(lg, -1).numpy()
            e.append(float((pg * np.arange(11)).sum()))
        eg_v.append(e)
        out = parse(from_model(dec[r["input"]]["greedy"]))
        greedy_grade.append(out.grade if out.grade is not None else 5)
        gold.append(V.index(r["output"].split("\n")[0].strip()))
logit3, eg_v, gold = np.array(logit3), np.array(eg_v), np.array(gold)
greedy_grade = np.array(greedy_grade, float)


def softmax(z, t=1.0):
    z = z / t
    z = z - z.max(1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(1, keepdims=True)


def ece(p, y, bins=10):
    conf, pred = p.max(1), p.argmax(1)
    out = 0.0
    for b in range(bins):
        m = (conf > b / bins) & (conf <= (b + 1) / bins)
        if m.any():
            out += m.mean() * abs((pred[m] == y[m]).mean() - conf[m].mean())
    return out


def nll(p, y):
    return float(-np.log(np.clip(p[np.arange(len(y)), y], 1e-12, 1)).mean())


def fit_t(z, y):
    ts = np.exp(np.linspace(math.log(0.3), math.log(5), 200))
    return float(ts[np.argmin([nll(softmax(z, t), y) for t in ts])])


rng = np.random.default_rng(0)
n = len(gold)
print(f"{run.name}: {n} rows")
# 1. calibration, one halving
idx = rng.permutation(n)
cal, test = idx[: n // 2], idx[n // 2 :]
t = fit_t(logit3[cal], gold[cal])
for name, pr in (("raw", softmax(logit3[test])), (f"temperature {t:.2f}", softmax(logit3[test], t))):
    print(f"calibration ({name}, test half): accuracy {100 * (pr.argmax(1) == gold[test]).mean():.1f}, "
          f"mean confidence {100 * pr.max(1).mean():.1f}, ECE {100 * ece(pr, gold[test]):.1f}, NLL {nll(pr, gold[test]):.3f}")
# 2. conformal sets, 50 halvings
for alpha in (0.10, 0.05):
    cov, size, single, single_acc = [], [], [], []
    for _ in range(50):
        idx = rng.permutation(n)
        cal, test = idx[: n // 2], idx[n // 2 :]
        t = fit_t(logit3[cal], gold[cal])
        pc, pt = softmax(logit3[cal], t), softmax(logit3[test], t)
        s = 1 - pc[np.arange(len(cal)), gold[cal]]
        q = np.quantile(s, min(1.0, math.ceil((len(cal) + 1) * (1 - alpha)) / len(cal)), method="higher")
        sets = (1 - pt) <= q
        cov.append(sets[np.arange(len(test)), gold[test]].mean())
        size.append(sets.sum(1).mean())
        one = sets.sum(1) == 1
        single.append(one.mean())
        single_acc.append((pt.argmax(1)[one] == gold[test][one]).mean() if one.any() else float("nan"))
    print(f"conformal {100 * (1 - alpha):.0f}%: coverage {100 * np.mean(cov):.1f}, mean set size {np.mean(size):.2f}, "
          f"single-verdict sets {100 * np.mean(single):.1f}% (accuracy {100 * np.nanmean(single_acc):.1f})")
# 3. ranking
p = softmax(logit3)
ordinal = np.array([RANK[V[g]] for g in gold])
e_rel = p[:, 0] + 0.5 * p[:, 1]
eg_top = eg_v[np.arange(n), p.argmax(1)]
eg = (p * eg_v).sum(1)


def concordance(score, label):
    ds, dl = np.sign(score[:, None] - score[None, :]), np.sign(label[:, None] - label[None, :])
    m = dl > 0
    return float(((ds[m] > 0) + 0.5 * (ds[m] == 0)).sum() / m.sum())


pred = p.argmax(1)
# the teacher's own grade for these rows (synth/tldr_grade.py on the realistic test), a finer label than the verdict
from relevance_explainer.synth.tldr_grade import key  # noqa: E402

tg = {}
for line in open("data/fmt_s/teacher.jsonl"):
    d = json.loads(line)
    if "grade" in d:
        tg[d["key"]] = d["grade"]
tgrade = np.array([tg.get(key(r), -1) for r in rows], float)
have = tgrade >= 0
print(f"teacher grade available for {have.sum()} rows")
for name, sc in (("greedy grade", greedy_grade), ("E[g]", eg), ("verdict probabilities", e_rel)):
    within = [concordance(sc[have & (gold == k)], tgrade[have & (gold == k)]) for k in range(3)]
    print(f"against the teacher grade, {name}: all {concordance(sc[have], tgrade[have]):.3f}; within teacher "
          f"relevant {within[0]:.3f}, partly {within[1]:.3f}, irrelevant {within[2]:.3f}")
for name, sc in (("greedy grade", greedy_grade), ("E[g | top verdict]", eg_top), ("E[g]", eg), ("verdict probabilities", e_rel)):
    within = [concordance(sc[pred == k], ordinal[pred == k]) for k in range(3)]
    print(f"ranking {name}: all {concordance(sc, ordinal):.3f}; within predicted relevant {within[0]:.3f}, "
          f"partly {within[1]:.3f}, irrelevant {within[2]:.3f}")
