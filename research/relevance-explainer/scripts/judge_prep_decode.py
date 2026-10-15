"""Write a run's decode-test answers (scripts/fmt_decode_test.py) as a generations file the judge reads
(tiny/judge.py: input, generated, reference), with the explanation in words only and the raw answer kept.

  python scripts/judge_prep_decode.py runs/fmt-s-C data/eval_real/itemfirst_nosim_v4 --col greedy --tag realnb
"""
import argparse
import json
from pathlib import Path

from relevance_explainer.claims import plain_output

ap = argparse.ArgumentParser()
ap.add_argument("run"); ap.add_argument("data_dir")
ap.add_argument("--col", default="greedy", choices=["greedy", "greedy_ban", "best"])
ap.add_argument("--tag", default="realnb")
a = ap.parse_args()
run = Path(a.run)
src = run / f"decode_test_{Path(a.data_dir).name}.jsonl"
lang = {}
for line in open(Path(a.data_dir) / "test.jsonl"):
    r = json.loads(line)
    lang[r["input"]] = r["language"]
with open(run / f"{a.tag}_generations_int8.jsonl", "w") as f:
    for line in open(src):
        d = json.loads(line)
        f.write(json.dumps({"input": d["input"], "generated": plain_output(d[a.col]), "reference": d["reference"],
                            "language": lang[d["input"]], "raw": d[a.col]}, ensure_ascii=False) + "\n")
print("wrote", run / f"{a.tag}_generations_int8.jsonl")
