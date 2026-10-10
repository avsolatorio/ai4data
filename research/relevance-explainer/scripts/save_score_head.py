"""Fit the 9-parameter answer-confidence head (target: verdict right and explanation good; inputs: the 8 uncertainty
scalars of tiny/selfcheck.py) on the selfcheck training features and save it as JSON for the app.

  python scripts/save_score_head.py runs/v2g-from-dapt-8ep-dpo/selfcheck
"""
import json
import sys
from pathlib import Path

import torch

from relevance_explainer.tiny.selfcheck import SCALARS, auroc, ece, fit, inputs_of, targets

O = Path(sys.argv[1])


def load(p):
    d = torch.load(p, weights_only=False)
    keep = [j for j, m in enumerate(d["meta"]) if m["judge"]]
    return {k: (v[keep] if torch.is_tensor(v) else [v[j] for j in keep]) for k, v in d.items() if k != "scalar_names"}


tr, te = load(O / "train_features.pt"), load(O / "test_features.pt")
x, mu, sd = inputs_of(tr, "scalars")
head = fit(x, targets(tr["meta"])["both"], [m["i"] for m in tr["meta"]], 0)
xt, _, _ = inputs_of(te, "scalars", mu, sd)
greedy = torch.tensor([m["k"] == 0 for m in te["meta"]])
with torch.no_grad():
    p = torch.sigmoid(head(xt))[greedy]
y = targets(te["meta"])["both"][greedy]
print(f"realistic test, greedy outputs: AUROC {auroc(p, y):.3f}  ECE {ece(p, y):.3f}")
lin = head.net
out = {"target": "verdict matches the teacher and the explanation is judged faithful >= 4, coherent >= 4, consistent, "
                  "in the right language, with no invented number",
       "features": SCALARS + ["invented_number"], "mean": mu.tolist(), "std": sd.tolist(),
       "weight": lin.weight[0].tolist(), "bias": lin.bias.item(), "test_auroc": auroc(p, y), "test_ece": ece(p, y)}
(O / "score_head.json").write_text(json.dumps(out, indent=1))
print(json.dumps({k: out[k] for k in ("features", "weight", "bias")}, indent=None))
