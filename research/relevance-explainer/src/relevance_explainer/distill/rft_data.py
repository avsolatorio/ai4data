"""Build a fine-tuning set from the chosen narratives of the preference pairs (rejection-sampling fine-tuning).

  python -m relevance_explainer.distill.rft_data --data data/f1 --out data/f1rft
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--pairs", default="dpo_pairs.jsonl")
    a = ap.parse_args()
    data, out = Path(a.data), Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    rows = [json.loads(l) for l in (data / "train.jsonl").read_text().splitlines()]
    n = 0
    with (out / "train.jsonl").open("w") as f:
        for line in (data / a.pairs).read_text().splitlines():
            p = json.loads(line)
            f.write(json.dumps({**rows[p["i"]], "output": f"{p['verdict']}\n{p['chosen']}"}, ensure_ascii=False) + "\n")
            n += 1
    for split in ("val", "test"):  # evaluation splits stay the original ones
        link = out / f"{split}.jsonl"
        if not link.exists():
            os.symlink(os.path.relpath(data / f"{split}.jsonl", out), link)
    print(f"{n} rows written to {out}/train.jsonl")


if __name__ == "__main__":
    main()
