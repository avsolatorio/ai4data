"""Decoding test for the format arms: does the repeat ban hurt structured outputs, and does resampling with the
verifier as the selector raise the share of fully verified answers?

For one run (int8) on one test file:
  1. greedy decoding with the 4-gram repeat ban (as in evaluation) and without it;
  2. K samples (temperature 0.7, top-k 40) without the ban, and selection: among the greedy answer and the samples
     with the greedy verdict (so the verdict, and verdict accuracy, stay as greedy decoding gives them), the first
     that is fully verified, else the one with the highest share of verified references and fewest unverified
     mentions.
"Fully verified" uses claims.verify: for a tagged run (C) on the model's own claims and references; for an untagged
run (A, B) on claims.annotate of the plain explanation, which links the spans that match record values, so only
mentions outside the record and search can fail. Reported per condition: verdict accuracy, mentions outside the
input, fully verified, and for C references verified and answers with a failed reference.

  CUDA_VISIBLE_DEVICES=0 python scripts/fmt_decode_test.py runs/fmt-s-C2 data/eval_real/itemfirst_nosim_v4/test.jsonl --k 6
"""
import argparse
import json
import time
from pathlib import Path

import torch
from tokenizers import Tokenizer

from relevance_explainer.claims import annotate, from_model, plain_output, sanitize_input, verify
from relevance_explainer.tiny.model import Config, TinyLM
from relevance_explainer.tiny.train import EOS, SEP, quantize

ap = argparse.ArgumentParser()
ap.add_argument("run"); ap.add_argument("data")
ap.add_argument("--k", type=int, default=6)
ap.add_argument("--limit", type=int, default=0)
ap.add_argument("--batch", type=int, default=48)
ap.add_argument("--max-new", type=int, default=420)
a = ap.parse_args()

run = Path(a.run)
ck = torch.load(run / "model_fp32.pt", map_location="cpu")
model = TinyLM(Config(**ck["config"]))
model.load_state_dict(ck["state"])
model, _ = quantize(model.cuda().eval(), 8)
tk = Tokenizer.from_file(str(run / "tokenizer.json"))
tagged = tk.token_to_id("<|exp|>") is not None
rows = [json.loads(line) for line in open(a.data)][: a.limit or None]
inputs = [sanitize_input(r["input"]) for r in rows]
gold = [r["output"].split("\n")[0].strip() for r in rows]
prompts = [tk.encode(x + SEP).ids for x in inputs]
order = sorted(range(len(prompts)), key=lambda i: len(prompts[i]))


def decode(no_repeat: int, temperature: float = 0.0, seed: int = 0) -> list[str]:
    torch.manual_seed(seed)
    out = [""] * len(prompts)
    for s in range(0, len(order), a.batch):
        ch = order[s : s + a.batch]
        with torch.autocast("cuda", dtype=torch.bfloat16):
            gs = model.generate_batch([prompts[i] for i in ch], EOS, a.max_new, no_repeat=no_repeat, temperature=temperature)
        for i, g in zip(ch, gs):
            out[i] = tk.decode(g, skip_special_tokens=not tagged).strip()
    return out


def check(text: str, inp: str):
    if tagged:
        return verify(from_model(text), inp)
    v, _, exp = plain_output(text).partition("\n")
    return verify(annotate(v.strip(), exp.strip(), inp), inp)


def score(ck) -> tuple:
    return (ck.all_verified, ck.verified_share if ck.verified_share is not None else 1.0, -len(ck.unverified))


def report(name: str, texts: list[str], cks: list) -> None:
    n = len(texts)
    acc = sum(t.split("\n")[0].strip() == g for t, g in zip(texts, gold)) / n
    line = (f"| {run.name} | {name} | {100 * acc:.1f} | {100 * sum(bool(c.unverified) for c in cks) / n:.1f} "
            f"| {100 * sum(c.all_verified for c in cks) / n:.1f} |")
    if tagged:
        refs = sum(len(c.refs) for c in cks)
        line += (f" {100 * sum(r.ok for c in cks for r in c.refs) / max(refs, 1):.1f} "
                 f"| {100 * sum(any(not r.ok for r in c.refs) for c in cks) / n:.1f} |")
    else:
        line += " - | - |"
    print(line, flush=True)


t0 = time.time()
greedy4 = decode(4)
report("greedy, repeat ban", greedy4, [check(t, x) for t, x in zip(greedy4, inputs)])
greedy0 = decode(0)
cks0 = [check(t, x) for t, x in zip(greedy0, inputs)]
report("greedy, no ban", greedy0, cks0)
best, best_ck = list(greedy0), list(cks0)
for k in range(a.k):
    samp = decode(0, temperature=0.7, seed=1000 + k)
    for i, (t, x) in enumerate(zip(samp, inputs)):
        if best_ck[i].all_verified or t.split("\n")[0].strip() != greedy0[i].split("\n")[0].strip():
            continue
        c = check(t, x)
        if score(c) > score(best_ck[i]):
            best[i], best_ck[i] = t, c
    report(f"best of greedy + {k + 1} samples", best, best_ck)
print(f"{time.time() - t0:.0f} s", flush=True)
with open(run / f"decode_test_{Path(a.data).parent.name}.jsonl", "w") as f:
    for r, g4, g0, b in zip(rows, greedy4, greedy0, best):
        f.write(json.dumps({"input": r["input"], "reference": r["output"], "greedy_ban": g4, "greedy": g0, "best": b},
                           ensure_ascii=False) + "\n")
