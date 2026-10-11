"""Calibration of the saved answer-confidence head (selfcheck/score_head.json) on greedy outputs of several sets.

  python scripts/calibration_report.py runs/v2g-from-dapt-8ep-dpo/selfcheck test challenge tables para

For each set: share fully correct, AUROC, ECE, share correct among the top 80% and 50% by score, the same for rows where
the teacher and Gemma 4 agree, and observed against predicted by score bin. Also per challenge phenomenon (when the
rows carry one): accuracy and mean score, to show confident systematic errors.
"""
import json
import sys
from collections import defaultdict
from pathlib import Path

import torch

from relevance_explainer.tiny.answer_score import ScoreHead
from relevance_explainer.tiny.selfcheck import auroc, ece, selective, targets

O = Path(sys.argv[1])
head = ScoreHead(O / "score_head.json")
DATA = {"test": "data/eval_real/itemfirst_nosim_v4", "challenge": "data/eval_challenge/itemfirst_nosim_v4",
        "tables": "data/eval_tables/itemfirst_nosim", "para": "data/eval_para/itemfirst_nosim_v4"}
out = {}
for name in sys.argv[2:]:
    F = torch.load(O / f"{name}_features.pt", weights_only=False)
    rows = [json.loads(l) for l in open(Path(DATA[name]) / "test.jsonl")]
    keep = [j for j, m in enumerate(F["meta"]) if m["k"] == 0 and m["judge"]]
    meta = [F["meta"][j] for j in keep]
    p = torch.tensor([head(F["scalars"][j].tolist()) for j in keep])
    y = targets(meta)
    agreed = torch.tensor([bool(rows[m["i"]].get("unanimous")) for m in meta])
    r = {"n": len(keep), "verdict_acc": y["verdict"].mean().item(), "explanation_good": y["explanation"].mean().item(),
         "both": y["both"].mean().item(), "auroc": auroc(p, y["both"]), "ece": ece(p, y["both"]),
         "top80": selective(p, y["both"], 0.8), "top50": selective(p, y["both"], 0.5), "mean_score": p.mean().item()}
    if agreed.any():
        pa, ya = p[agreed], y["both"][agreed]
        r["agreed"] = {"n": int(agreed.sum()), "both": ya.mean().item(), "auroc": auroc(pa, ya), "ece": ece(pa, ya)}
    r["bins"] = []
    for lo, hi in ((0, 0.5), (0.5, 0.7), (0.7, 0.8), (0.8, 0.9), (0.9, 1.01)):
        m = (p >= lo) & (p < hi)
        if m.any():
            r["bins"].append({"range": f"{lo:.1f}-{min(hi, 1):.1f}", "n": int(m.sum()), "predicted": p[m].mean().item(),
                              "observed": y["both"][m].mean().item()})
    phen = defaultdict(list)
    for j, m in enumerate(meta):
        key = rows[m["i"]].get("relation") or rows[m["i"]].get("phenomenon") or rows[m["i"]].get("mode")
        if key:
            phen[key].append(j)
    if len(phen) > 1:
        r["by_group"] = {k: {"n": len(js), "both": y["both"][js].mean().item(), "verdict": y["verdict"][js].mean().item(),
                             "mean_score": p[js].mean().item(),
                             "confident_wrong": ((p[js] >= 0.8) & (y["both"][js] == 0)).float().mean().item()}
                         for k, js in sorted(phen.items())}
    out[name] = r
    print(f"\n== {name}: n {r['n']}  verdict {r['verdict_acc']:.3f}  explanation good {r['explanation_good']:.3f}  "
          f"both {r['both']:.3f}  mean score {r['mean_score']:.3f}")
    print(f"   AUROC {r['auroc']:.3f}  ECE {r['ece']:.3f}  correct in top 80% {r['top80']:.3f}, top 50% {r['top50']:.3f}")
    if "agreed" in r:
        a = r["agreed"]
        print(f"   teachers agree: n {a['n']}  both {a['both']:.3f}  AUROC {a['auroc']:.3f}  ECE {a['ece']:.3f}")
    for b in r["bins"]:
        print(f"   score {b['range']}: predicted {b['predicted']:.3f}  observed {b['observed']:.3f}  (n={b['n']})")
    for k, g in (r.get("by_group") or {}).items():
        print(f"   {k:18s} n {g['n']:4d}  both {g['both']:.3f}  verdict {g['verdict']:.3f}  mean score {g['mean_score']:.3f}  "
              f"score>=0.8 but wrong {g['confident_wrong']:.3f}")
(O / "calibration_report.json").write_text(json.dumps(out, indent=1))
