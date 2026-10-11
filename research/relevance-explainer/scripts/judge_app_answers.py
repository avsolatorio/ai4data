"""Judge the app's final answers (scripts/app_answers.py) with the teacher, and for answers that were retried also the
first answer, so the retry's effect is measured on the deployed pipeline. Reports accuracy and calibration of the
answer confidence on the app's own outputs.

  TEACHER_URLS=... python scripts/judge_app_answers.py runs/v2g-from-dapt-8ep-dpo/app_answers_realtest.jsonl \\
      data/eval_real/itemfirst_nosim_v4
"""
import json
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from relevance_explainer.distill.onpolicy import score_one
from relevance_explainer.distill.teacher_checks import urls
from relevance_explainer.tiny.numguard import unsupported

path, data = Path(sys.argv[1]), Path(sys.argv[2])
out = path.with_name(path.stem + "_judged.jsonl")
rows = [json.loads(l) for l in (data / "test.jsonl").open()]
ans = [json.loads(l) for l in path.open()]
ans = [a for a in ans if a["event"] and a["event"].get("type") == "done"]
if not out.exists():
    servers = urls()
    jobs = []
    for a in ans:
        ev = a["event"]
        jobs.append((a["i"], "final", ev["verdict"], ev["narrative"]))
        if ev["stats"].get("retried"):
            jobs.append((a["i"], "first", ev["verdict"], ev["attempts"][0]["narrative"]))
    with ThreadPoolExecutor(96) as ex:
        res = list(ex.map(lambda j: score_one(servers[j[0] % len(servers)], rows[j[0]]["input"], j[2], j[3]), jobs))
    with out.open("w") as f:
        for (i, kind, v, n), s in zip(jobs, res):
            f.write(json.dumps({"i": i, "kind": kind, "verdict": v, "narrative": n, "judge": s}, ensure_ascii=False) + "\n")
J = {(d["i"], d["kind"]): d for d in map(json.loads, out.open())}


def good(i, kind):
    d = J.get((i, kind))
    s = d and d["judge"]
    if not s:
        return None
    ok = s["faithful"] >= 4 and s["coherent"] >= 4 and s["consistent"] and s["language_ok"]
    return ok and not unsupported(d["verdict"] + "\n" + d["narrative"], rows[i]["input"])


fin = [(a, good(a["i"], "final")) for a in ans]
fin = [(a, g) for a, g in fin if g is not None]
n = len(fin)
v = sum(a["event"]["verdict"] == a["label"] for a, _ in fin) / n
e = sum(g for _, g in fin) / n
both = [(a["event"]["verdict"] == a["label"]) and g for a, g in fin]
print(f"app answers judged: {n}; verdict {v:.3f}; explanation good {e:.3f}; fully correct {sum(both) / n:.3f}")
ret = [a for a, _ in fin if a["event"]["stats"].get("retried")]
pairs = [(good(a["i"], "first"), good(a["i"], "final")) for a in ret]
pairs = [p for p in pairs if None not in p]
changed = sum(1 for a in ret if not a["event"]["attempts"][0]["kept"])
print(f"retried {len(ret)} ({len(ret) / n:.1%}); a rewrite was kept in {changed}; explanation good: first answer "
      f"{sum(p[0] for p in pairs) / max(1, len(pairs)):.3f} -> final {sum(p[1] for p in pairs) / max(1, len(pairs)):.3f} "
      f"(n={len(pairs)})")
print("answer confidence against fully correct (app outputs):")
for lo, hi in ((0, 0.5), (0.5, 0.7), (0.7, 0.8), (0.8, 0.9), (0.9, 1.01)):
    b = [(a["event"]["score"], c) for (a, _), c in zip(fin, both) if lo <= a["event"]["score"] < hi]
    if b:
        print(f"  {lo:.1f}-{min(hi, 1):.1f}: predicted {sum(x for x, _ in b) / len(b):.3f}  observed "
              f"{sum(c for _, c in b) / len(b):.3f}  (n={len(b)})")
