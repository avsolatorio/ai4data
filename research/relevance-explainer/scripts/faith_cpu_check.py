"""The faithfulness scores the app computed on CPU (scripts/cpu_parity_c.py) against the judges' labels, on the rows
where the CPU answer is identical to the judged GPU greedy answer (judge labels exist only for that text).

  python scripts/faith_cpu_check.py runs/final-C-dpo
"""
import json
import random
import sys
from pathlib import Path

import numpy as np
from sklearn.metrics import roc_auc_score

run = Path(sys.argv[1])
cpu = {json.loads(l)["input"]: json.loads(l) for l in open(run / "cpu_answers_itemfirst_nosim_v4.jsonl")}
gen = [json.loads(l) for l in open(run / "realnb_generations_int8.jsonl")]
for suf, name in (("", "teacher"), ("_gemma4", "Gemma 4")):
    j = json.loads((run / f"judge_realnb_int8{suf}.json").read_text())
    rows = random.Random(0).sample(gen, max(x["k"] for x in j["items"]) + 1)
    lab = {x["k"]: x["score"]["faithful"] >= 4 for x in j["items"] if x["who"] == "student" and x["score"]}
    p, y = [], []
    for k, ok in lab.items():
        c = cpu.get(rows[k]["input"])
        if c and c["faith"] and c["text"] == rows[k]["raw"].strip():
            p.append(c["faith"]["score"]); y.append(ok)
    p, y = np.array(p), np.array(y, int)
    hi = p >= 0.8
    print(f"{name}: {len(y)} identical answers, AUROC {roc_auc_score(y, p):.3f}; score 0.8+ {100 * hi.mean():.1f}% of answers, "
          f"judge 4-5 {100 * y[hi].mean():.1f}% above and {100 * y[~hi].mean():.1f}% below")
