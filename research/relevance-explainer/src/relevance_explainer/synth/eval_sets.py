"""Evaluation directories for the realistic test set, one per input format.

From data/real_test/examples_labeled.jsonl (consensus labels) writes data/eval_real/{full,min,itemfirst}/test.jsonl
(and an identical val.jsonl and a one-line train.jsonl so the training tools can read the directory), with the
stored teacher label as the target and the three labels kept for scoring.

  python -m relevance_explainer.synth.eval_sets
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from .realistic import item_first


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default="data/real_test/examples_labeled.jsonl")
    ap.add_argument("--out", default="data/eval_real")
    a = ap.parse_args()
    rows = [json.loads(l) for l in Path(a.src).read_text().splitlines()]
    for name, key in (("full", "input_v2"), ("min", "input_v2min"), ("itemfirst", None)):
        d = Path(a.out) / name
        d.mkdir(parents=True, exist_ok=True)
        out = []
        for r in rows:
            inp = item_first(r["input_v2"]) if key is None else r[key]
            out.append({"input": inp, "output": f"{r['verdict']}\n{r['narrative']}", "verdict": r["verdict"], "language": r["language"],
                        "source": r["record"]["kind"], "labels": r["labels"], "majority": r["majority"], "unanimous": r["unanimous"],
                        "style": r.get("style"), "mode": r.get("mode"), "relation": r.get("relation")})
        text = "".join(json.dumps(x, ensure_ascii=False) + "\n" for x in out)
        (d / "test.jsonl").write_text(text)
        (d / "val.jsonl").write_text(text)
        (d / "train.jsonl").write_text(text.split("\n", 1)[0] + "\n")
        print(name, len(out))


if __name__ == "__main__":
    main()
