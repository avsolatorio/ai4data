"""Does the model's grade rank results better than its verdict probabilities?

For a tagged run on the realistic test: the grade from its greedy answers (decode test file) and the expected relevance
from the verdict probabilities at the first output step, E = P(relevant) + 0.5 P(partly_relevant) (int8, as in the
app). Each score is compared with the teacher verdict as an ordinal label (irrelevant < partly_relevant < relevant) by
pairwise concordance: the share of pairs with different labels that the score orders correctly (ties count half).
Reported overall, and within each predicted verdict (where only the extra signal can order pairs), for the grade, E,
and the grade with E as a tie-breaker.

  CUDA_VISIBLE_DEVICES=0 python scripts/grade_vs_probs.py runs/fmt-s-C data/eval_real/itemfirst_nosim_v4/test.jsonl
"""
import json
import sys
from pathlib import Path

import numpy as np
import torch
from tokenizers import Tokenizer

from relevance_explainer.claims import from_model, parse, sanitize_input
from relevance_explainer.tiny.model import Config, TinyLM
from relevance_explainer.tiny.train import SEP, quantize
from relevance_explainer.viewer import verdict_probs

run, data = Path(sys.argv[1]), sys.argv[2]
ck = torch.load(run / "model_fp32.pt", map_location="cpu")
model = TinyLM(Config(**ck["config"]))
model.load_state_dict(ck["state"])
model, _ = quantize(model.cuda().eval(), 8)
tk = Tokenizer.from_file(str(run / "tokenizer.json"))
rows = [json.loads(line) for line in open(data)]
dec = {json.loads(line)["input"]: json.loads(line) for line in open(run / f"decode_test_{Path(data).parent.name}.jsonl")}
RANK = {"irrelevant": 0, "partly_relevant": 1, "relevant": 2}

gold, pred, grade, e = [], [], [], []
with torch.no_grad():
    for r in rows:
        ids = torch.tensor([tk.encode(sanitize_input(r["input"]) + SEP).ids], device="cuda")
        prefix = torch.tensor([ids.size(1)], device="cuda") if model.c.prefix_lm else None
        with torch.autocast("cuda", dtype=torch.bfloat16):
            logits = model(ids, prefix=prefix)[0, -1]
        p = verdict_probs(tk, logits)
        out = parse(from_model(dec[r["input"]]["greedy"]))
        gold.append(RANK.get(r["output"].split("\n")[0].strip(), 1))
        pred.append(out.verdict)
        grade.append(out.grade if out.grade is not None else 5)
        e.append(p["relevant"] + 0.5 * p["partly_relevant"])
gold, grade, e, pred = np.array(gold), np.array(grade, float), np.array(e), np.array(pred)


def concordance(score: np.ndarray, label: np.ndarray) -> tuple[float, int]:
    ds, dl = np.sign(score[:, None] - score[None, :]), np.sign(label[:, None] - label[None, :])
    m = dl > 0
    n = int(m.sum())
    return (float(((ds[m] > 0) + 0.5 * (ds[m] == 0)).sum() / n) if n else float("nan")), n


scores = {"grade": grade, "verdict probabilities": e, "grade, then probabilities": grade + 0.01 * e}
out = {}
for name, idx in [("all", np.ones(len(gold), bool))] + [(f"predicted {v}", pred == v) for v in RANK]:
    out[name] = {k: concordance(s[idx], gold[idx]) for k, s in scores.items()}
    print(name, {k: f"{v[0]:.3f} ({v[1]} pairs)" for k, v in out[name].items()}, flush=True)
(run / "grade_vs_probs.json").write_text(json.dumps(out, indent=1))
