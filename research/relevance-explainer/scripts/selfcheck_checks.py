"""Checks on the correctness-head result (tiny/selfcheck.py):
1. in distribution: train on 80% of the validation inputs, test on the other 20% (greedy outputs), so a loss of the
   hidden-state heads on the realistic test can be told apart from distribution shift;
2. retry on the realistic test with each head kind for the explanation and for both targets;
3. the 9-parameter head's calibration table on the realistic test (target both).

  python scripts/selfcheck_checks.py runs/v2g-from-dapt-8ep-dpo/selfcheck
"""
import random
import sys
from pathlib import Path

import torch

from relevance_explainer.tiny.selfcheck import fit, inputs_of, report, targets

O = Path(sys.argv[1])
torch.manual_seed(0)


def load(p):
    d = torch.load(p, weights_only=False)
    keep = [j for j, m in enumerate(d["meta"]) if m["judge"]]
    return {k: (v[keep] if torch.is_tensor(v) else [v[j] for j in keep]) for k, v in d.items() if k != "scalar_names"}


def subset(d, idx):
    return {k: (v[idx] if torch.is_tensor(v) else [v[j] for j in idx]) for k, v in d.items()}


tr, te = load(O / "train_features.pt"), load(O / "test_features.pt")
KINDS = (("scalars", 0), ("hidden", 64), ("hidden+scalars", 64))

print("== 1. in distribution: 80/20 split of the validation inputs (greedy outputs of the 20%)")
inputs = sorted({m["i"] for m in tr["meta"]})
held = set(random.Random(1).sample(inputs, len(inputs) // 5))
a = [j for j, m in enumerate(tr["meta"]) if m["i"] not in held]
b = [j for j, m in enumerate(tr["meta"]) if m["i"] in held and m["k"] == 0]
A, B = subset(tr, a), subset(tr, b)
ya, yb = targets(A["meta"]), targets(B["meta"])
for target in ("verdict", "explanation", "both"):
    print(f" target {target}")
    report("verdict probability", B["scalars"][:, 0], yb[target])
    for kind, h in KINDS:
        xa, mu, sd = inputs_of(A, kind)
        xb, _, _ = inputs_of(B, kind, mu, sd)
        head = fit(xa, ya[target], [m["i"] for m in A["meta"]], h)
        with torch.no_grad():
            report(f"{kind} {h}", torch.sigmoid(head(xb)), yb[target])

print("\n== 2. retry on the realistic test: keep the highest-scoring of greedy + 3 samples")
ytr, yte = targets(tr["meta"]), targets(te["meta"])
by = {}
for j, m in enumerate(te["meta"]):
    by.setdefault(m["i"], []).append(j)
greedy = [next(j for j in js if te["meta"][j]["k"] == 0) for js in by.values() if any(te["meta"][j]["k"] == 0 for j in js)]
f = lambda js: sum(te["meta"][j]["judge"]["faithful"] for j in js) / len(js)  # noqa: E731
print(f"  greedy: good {yte['explanation'][greedy].mean():.3f} faithful {f(greedy):.3f}")
probs = {}
for target in ("explanation", "both"):
    for kind, h in KINDS:
        xtr, mu, sd = inputs_of(tr, kind)
        xte, _, _ = inputs_of(te, kind, mu, sd)
        head = fit(xtr, ytr[target], [m["i"] for m in tr["meta"]], h)
        with torch.no_grad():
            p = torch.sigmoid(head(xte))
        probs[(target, kind)] = p
        best = [max(js, key=lambda j: p[j].item()) for js in by.values() if len(js) > 1]
        print(f"  head {target:11s} {kind:15s}: good {yte['explanation'][best].mean():.3f} faithful {f(best):.3f}")
lp = te["scalars"][:, 1]
best = [max(js, key=lambda j: lp[j].item()) for js in by.values() if len(js) > 1]
print(f"  mean token log-prob only (no training): good {yte['explanation'][best].mean():.3f} faithful {f(best):.3f}")

print("\n== 3. calibration of the 9-parameter head, target both, realistic test greedy outputs")
p = probs[("both", "scalars")][greedy]
y = yte["both"][greedy]
for lo, hi in ((0, 0.5), (0.5, 0.7), (0.7, 0.8), (0.8, 0.9), (0.9, 1.01)):
    m = (p >= lo) & (p < hi)
    if m.any():
        print(f"  {lo:.1f}-{min(hi, 1):.1f}: predicted {p[m].mean():.3f}  observed {y[m].mean():.3f}  (n={int(m.sum())})")
