"""Manifest of the artifacts behind the released claim-carrying model (REPRODUCE.md): for each file, its SHA-256, size
and, for JSONL, the number of rows. A rebuilt or downloaded artifact is checked against it with --check.

  python scripts/manifest.py                 # write artifacts.json
  python scripts/manifest.py --check         # compare the files present with artifacts.json
"""
import argparse
import hashlib
import json
import sys
from pathlib import Path

ARTIFACTS = {
    "corpus": ["data/corpus_multi_final/tokenizer.json", "data/corpus_multi_final/plan_target.json"],
    "base": ["runs/base-final/state-457764.pt", "runs/base-2k/model.pt"],
    "task data (round G, no similarity line)": [f"data/v2_g_itemfirst_nosim/{s}.jsonl" for s in ("train", "val", "test")],
    "teacher tl;dr and grade": ["data/fmt_s/teacher.jsonl", "data/fmt/teacher_rest.jsonl"],
    "format C data": [f"data/final/C/{s}.jsonl" for s in ("train", "val", "test")],
    "fine-tuned model": ["runs/final-C/model_fp32.pt", "runs/final-C/tokenizer.json"],
    "DPO round": ["data/final/C/onpolicy_c_final-C_samples.jsonl", "data/final/C/onpolicy_c_final-C_scored.jsonl",
                  "data/final/C/dpo_c_pairs_final-C.jsonl", "runs/final-C-dpo/model_fp32.pt", "runs/final-C-dpo/dpo_log.json"],
    "serving files": ["runs/final-C-dpo/explainer-int8.pt", "runs/final-C-dpo/calibration.json",
                      "runs/final-C-dpo/faith_scorer.json", "data/final/C/faith_final-C-dpo_s1_samples.jsonl",
                      "data/final/C/faith_final-C-dpo_s1_scored.jsonl"],
    "test sets": ["data/eval_real/itemfirst_nosim_v4/test.jsonl", "data/eval_para/itemfirst_nosim_v4/test.jsonl",
                  "data/schemas_eval2/itemfirst_nosim_v4/test.jsonl", "data/eval_challenge/itemfirst_nosim_v4/test.jsonl",
                  "data/eval_tables/itemfirst_nosim/test.jsonl"],
    "third-party": ["data/geo/lid.176.ftz"],
}


def describe(p: Path) -> dict:
    h, n, rows = hashlib.sha256(), 0, 0
    with open(p, "rb") as f:
        for block in iter(lambda: f.read(1 << 22), b""):
            h.update(block)
            n += len(block)
            rows += block.count(b"\n") if p.suffix == ".jsonl" else 0
    return {"sha256": h.hexdigest(), "bytes": n, **({"rows": rows} if p.suffix == ".jsonl" else {})}


ap = argparse.ArgumentParser()
ap.add_argument("--check", action="store_true")
ap.add_argument("--out", default="artifacts.json")
a = ap.parse_args()
if a.check:
    ref = json.loads(Path(a.out).read_text())
    bad = 0
    for group, files in ref["artifacts"].items():
        for path, want in files.items():
            if not Path(path).exists():
                print(f"absent    {path}")
                continue
            got = describe(Path(path))
            ok = got["sha256"] == want["sha256"]
            bad += not ok
            print(f"{'ok' if ok else 'DIFFERS':9s} {path}")
    sys.exit(1 if bad else 0)
out = {"note": "SHA-256, size and JSONL rows of the files behind runs/final-C-dpo; see REPRODUCE.md", "artifacts": {}}
for group, files in ARTIFACTS.items():
    out["artifacts"][group] = {}
    for f in files:
        p = Path(f)
        out["artifacts"][group][f] = describe(p) if p.exists() else {"missing": True}
        print(f, "missing" if not p.exists() else out["artifacts"][group][f]["sha256"][:12], flush=True)
Path(a.out).write_text(json.dumps(out, indent=1) + "\n")
