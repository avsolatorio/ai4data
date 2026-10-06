"""Paraphrase test set: anchored searches (every fully-answering one written away from the record's terms) narrated by
the teacher, with a Gemma 4 second verdict, written in the item-first formats with and without the similarity line.

  python -m relevance_explainer.synth.para_eval --src data/para_test --out data/eval_para
"""
from __future__ import annotations

import argparse
import json
import re
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from . import llm
from .realistic import NARRATE_V2, item_first

GEMMA = ["http://localhost:8002/v1"]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default="data/para_test")
    ap.add_argument("--out", default="data/eval_para")
    ap.add_argument("--workers", type=int, default=16)
    a = ap.parse_args()
    rows = [json.loads(l) for l in (Path(a.src) / "examples.jsonl").read_text().splitlines()]

    def gemma(r):
        try:
            return llm.parse_json(llm.chat("gemma-4-31b", NARRATE_V2, r["input_v2"], temperature=0.0, max_tokens=450, base_urls=GEMMA)[0]).get("verdict")
        except Exception:
            return None

    with ThreadPoolExecutor(a.workers) as ex:
        g = list(ex.map(gemma, rows))
    for name, strip in (("itemfirst", False), ("itemfirst_nosim", True)):
        d = Path(a.out) / name
        d.mkdir(parents=True, exist_ok=True)
        out = []
        for r, gv in zip(rows, g):
            inp = item_first(r["input_v2"])
            if strip:
                inp = re.sub(r"\nRetrieval similarity: [-0-9.]+", "", inp)
            out.append({"input": inp, "output": f"{r['verdict']}\n{r['narrative']}", "verdict": r["verdict"], "language": r["language"],
                        "source": r["record"]["kind"], "labels": {"stored": r["verdict"], "gemma4": gv}, "majority": r["verdict"],
                        "unanimous": gv == r["verdict"], "style": r.get("style"), "mode": r.get("wording", "free"),
                        "relation": r.get("relation")})
        text = "".join(json.dumps(x, ensure_ascii=False) + "\n" for x in out)
        for split in ("test", "val"):
            (d / f"{split}.jsonl").write_text(text)
        (d / "train.jsonl").write_text(text.split("\n", 1)[0] + "\n")
    agree = sum(gv == r["verdict"] for r, gv in zip(rows, g)) / max(1, len(rows))
    print(f"paraphrase test set: {len(rows)} examples; Gemma 4 agrees with the teacher on {agree:.3f}")


if __name__ == "__main__":
    main()
