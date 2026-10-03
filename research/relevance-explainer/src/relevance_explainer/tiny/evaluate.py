"""Evaluate a saved task model (fp32, INT8, INT4) on a test split with batched KV-cache decoding.

  CUDA_VISIBLE_DEVICES=1 python -m relevance_explainer.tiny.evaluate --run runs/task-x --data data/v2
"""
from __future__ import annotations

import argparse
import json
import random
import time
from pathlib import Path

import torch
from tokenizers import Tokenizer

from ..train.finetune import read
from .model import Config, TinyLM
from .train import encode, evaluate, quantize


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--data", required=True)
    ap.add_argument("--split", default="test")
    ap.add_argument("--limit", type=int, default=0, help="evaluate a fixed random subset of this size")
    ap.add_argument("--precisions", nargs="*", default=["fp32", "int8", "int4"])
    ap.add_argument("--max-new", type=int, default=260)
    ap.add_argument("--batch", type=int, default=48)
    ap.add_argument("--tag", default="eval")
    ap.add_argument("--copy-bias", type=float, default=0.0, help="logit bonus for tokens that occur in the input")
    ap.add_argument("--script-guard", action="store_true", help="block tokens of scripts absent from the language and the input")
    a = ap.parse_args()

    run = Path(a.run)
    ck = torch.load(run / "model_fp32.pt", map_location="cpu")
    model = TinyLM(Config(**ck["config"]))
    model.load_state_dict(ck["state"])
    model = model.cuda().eval()
    tk = Tokenizer.from_file(str(run / "tokenizer.json"))

    rows = read(Path(a.data) / f"{a.split}.jsonl")
    if a.limit and len(rows) > a.limit:
        rows = random.Random(0).sample(rows, a.limit)
    enc, dropped = encode(tk, rows, model.c.max_len)
    print(f"{len(enc)} examples ({dropped} dropped as too long)", flush=True)

    out = {}
    for name in a.precisions:
        bits = {"fp32": None, "int8": 8, "int4": 4}[name]
        m, size = (model, model.n_params() * 4) if bits is None else quantize(model, bits)
        t = time.time()
        metrics, gens = evaluate(m, tk, enc, a.max_new, a.batch, a.copy_bias, a.script_guard)
        metrics["size_mb"] = round(size / 1e6, 2)
        metrics["eval_seconds"] = round(time.time() - t, 1)
        out[name] = metrics
        with (run / f"{a.tag}_generations_{name}.jsonl").open("w") as f:
            for g in gens:
                f.write(json.dumps(g, ensure_ascii=False) + "\n")
        print(name, {k: metrics[k] for k in ("size_mb", "verdict_accuracy", "invalid_verdict_rate", "eval_seconds")},
              "macroF1", round(metrics["f1"]["macro"], 3), flush=True)
    (run / f"{a.tag}_metrics.json").write_text(json.dumps(out, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
