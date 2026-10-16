"""The app's answer path (answer_c.answer: int8 file on CPU, greedy without the repeat ban, one round of verifier-
selected retries, calibration, expected grade, faithfulness score) on a test set, compared with the GPU decoding test
(scripts/fmt_decode_test.py). Writes <run>/cpu_answers_<data>.jsonl (resumable) and prints verdict accuracy, the
share of fully verified answers before and after retries, the conformal set coverage, the faithfulness bands and the
time per answer.

  python scripts/cpu_parity_c.py runs/final-C-dpo data/eval_real/itemfirst_nosim_v4/test.jsonl [--threads 4]
"""
import argparse
import json
from pathlib import Path

import numpy as np
import torch

from relevance_explainer.answer_c import Assets, answer
from relevance_explainer.claims import from_model, verify
from relevance_explainer.tiny.export import load_int8
from relevance_explainer.tiny.scriptguard import banned_for, token_masks

ap = argparse.ArgumentParser()
ap.add_argument("run"); ap.add_argument("data"); ap.add_argument("--threads", type=int, default=4)
a = ap.parse_args()
torch.set_num_threads(a.threads)
run = Path(a.run)
out = run / f"cpu_answers_{Path(a.data).parent.name}.jsonl"
rows = [json.loads(l) for l in open(a.data)]
done = {json.loads(l)["input"] for l in open(out)} if out.exists() else set()
model, tk = load_int8(run / "explainer-int8.pt", "cpu")
assets = Assets.from_run(model, tk, run)
masks = token_masks(tk)
with out.open("a", buffering=1) as f:
    for r in rows:
        if r["input"] in done:
            continue
        o = answer(assets, r["input"], banned=banned_for(masks, r["input"]))
        f.write(json.dumps({"input": r["input"], "gold": r["output"].split("\n")[0].strip(), "verdict": o["verdict"],
                            "probs": o["probs"], "verdict_set": o["verdict_set"], "expected_grade": o["expected_grade"],
                            "first_verified": o["attempts"][0]["all_verified"], "verified": o["check"]["all_verified"],
                            "faith": o["faith"], "retried": o["retried"], "text": o["text"], "stats": o["stats"]},
                           ensure_ascii=False) + "\n")
res = [json.loads(l) for l in open(out)]
n = len(res)
print(f"{n} answers ({run.name}, CPU int8, {a.threads} threads)")
print(f"verdict accuracy {100 * np.mean([x['verdict'] == x['gold'] for x in res]):.1f}")
print(f"fully verified: first answer {100 * np.mean([x['first_verified'] for x in res]):.1f}, after retries "
      f"{100 * np.mean([x['verified'] for x in res]):.1f} (retried {np.mean([x['retried'] > 0 for x in res]) * 100:.1f}% of answers)")
s = [x for x in res if x["verdict_set"] is not None]
if s:
    print(f"conformal set: coverage {100 * np.mean([x['gold'] in x['verdict_set'] for x in s]):.1f}, mean size "
          f"{np.mean([len(x['verdict_set']) for x in s]):.2f}, single {100 * np.mean([len(x['verdict_set']) == 1 for x in s]):.1f}")
f_ = [x["faith"]["band"] for x in res if x["faith"]]
if f_:
    print("faithfulness bands: " + ", ".join(f"{b} {100 * f_.count(b) / len(f_):.1f}%" for b in ("high", "check", "low")))
t = np.array([x["stats"]["total_s"] for x in res])
print(f"seconds per answer: median {np.median(t):.2f}, 90th percentile {np.percentile(t, 90):.2f}, max {t.max():.2f}")
gpu = run / f"decode_test_{Path(a.data).parent.name}.jsonl"
if gpu.exists():
    g = {json.loads(l)["input"]: json.loads(l)["greedy"] for l in open(gpu)}
    same = [g[x["input"]].strip() == x["text"] for x in res if x["input"] in g and x["retried"] == 0]
    gv = [verify(from_model(g[x["input"]]), x["input"]).all_verified for x in res if x["input"] in g]
    print(f"GPU greedy: fully verified {100 * np.mean(gv):.1f}; identical to the CPU answer for {100 * np.mean(same):.1f}% "
          f"of the answers that were not retried")
