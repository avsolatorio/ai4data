"""Agreement between the teacher judge and Gemma 4 on the same student answers (tiny/judge.py files
judge_<tag>_int8.json and judge_<tag>_int8_gemma4.json): agreement on "faithful 4 or 5", counts below 4 for each judge
and both, and the AUROC of the teacher's 1-5 score for Gemma's label (a ceiling reference for scorers fitted to the
teacher).

  python scripts/judge_agreement.py runs/final-B runs/final-C runs/final-C-dpo
"""
import argparse
import json

from sklearn.metrics import roc_auc_score

ap = argparse.ArgumentParser()
ap.add_argument("runs", nargs="+"); ap.add_argument("--tag", default="realnb")
a = ap.parse_args()


def scores(path):
    return {x["k"]: x["score"]["faithful"] for x in json.load(open(path))["items"] if x["who"] == "student" and x["score"]}


for r in a.runs:
    q, g = scores(f"{r}/judge_{a.tag}_int8.json"), scores(f"{r}/judge_{a.tag}_int8_gemma4.json")
    ks = sorted(q.keys() & g.keys())
    qa, ga = [q[k] >= 4 for k in ks], [g[k] >= 4 for k in ks]
    print(f"{r}: {len(ks)} answers, agree on 4-5 {sum(x == y for x, y in zip(qa, ga)) / len(ks):.3f}; below 4: teacher "
          f"{qa.count(False)}, Gemma {ga.count(False)}, both {sum(not x and not y for x, y in zip(qa, ga))}; "
          f"AUROC of the teacher's score for Gemma's label {roc_auc_score(ga, [q[k] for k in ks]):.3f}")
