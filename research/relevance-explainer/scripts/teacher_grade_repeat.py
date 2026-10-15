"""How reliable is the teacher's 0-10 grade? Re-grade a sample of realistic-test rows with the same prompt and checks
(synth/tldr_grade.one) and compare with the grade written for the training data (data/fmt_s/teacher.jsonl): exact
agreement, agreement within 1, mean absolute difference, and pairwise concordance between the two gradings, overall
and within each verdict (the grade's own information).

  LLM_EXTRA='{"chat_template_kwargs":{"enable_thinking":false}}' python scripts/teacher_grade_repeat.py --n 600
"""
import argparse
import json
import random
from concurrent.futures import ThreadPoolExecutor

import numpy as np

from relevance_explainer.synth.tldr_grade import key, one

ap = argparse.ArgumentParser()
ap.add_argument("--data", default="data/eval_real/itemfirst_nosim_v4/test.jsonl")
ap.add_argument("--teacher", default="data/fmt_s/teacher.jsonl")
ap.add_argument("--n", type=int, default=600)
ap.add_argument("--workers", type=int, default=48)
ap.add_argument("--out", default="data/fmt_s/teacher_grade_repeat.jsonl")
a = ap.parse_args()

old = {}
for line in open(a.teacher):
    d = json.loads(line)
    if "grade" in d:
        old[d["key"]] = d["grade"]
rows = [r for r in (json.loads(line) for line in open(a.data)) if key(r) in old]
rows = random.Random(0).sample(rows, min(a.n, len(rows)))
with ThreadPoolExecutor(a.workers) as ex:
    res = list(ex.map(one, rows))
with open(a.out, "w") as f:
    for r, x in zip(rows, res):
        f.write(json.dumps({"key": key(r), "old": old[key(r)], "new": x.get("grade")}, ensure_ascii=False) + "\n")
pairs = [(old[key(r)], x["grade"], r["output"].split("\n")[0].strip()) for r, x in zip(rows, res) if "grade" in x]
g1, g2 = np.array([p[0] for p in pairs], float), np.array([p[1] for p in pairs], float)
verdict = np.array([p[2] for p in pairs])


def concordance(s, label):
    ds, dl = np.sign(s[:, None] - s[None, :]), np.sign(label[:, None] - label[None, :])
    m = dl > 0
    return float(((ds[m] > 0) + 0.5 * (ds[m] == 0)).sum() / m.sum()) if m.any() else float("nan")


print(f"{len(pairs)} rows re-graded: exact {100 * (g1 == g2).mean():.1f}%, within 1 {100 * (abs(g1 - g2) <= 1).mean():.1f}%, "
      f"mean |diff| {abs(g1 - g2).mean():.2f}, concordance {concordance(g2, g1):.3f}")
for v in ("relevant", "partly_relevant", "irrelevant"):
    m = verdict == v
    print(f"  within {v} (n {m.sum()}): exact {100 * (g1[m] == g2[m]).mean():.1f}%, concordance {concordance(g2[m], g1[m]):.3f}")
