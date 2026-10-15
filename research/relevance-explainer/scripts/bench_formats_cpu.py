"""CPU time per answer for format runs, with the app's settings (int8, 4 threads, KV-cache decoding): one greedy
answer, and the greedy answer plus 3 samples written as one batch (the retry round). Tagged runs decode without the
repeat ban, plain runs with it. Same 40 realistic-test inputs for every run.

  python scripts/bench_formats_cpu.py runs/fmt-s-A runs/fmt-s-C
"""
import json
import random
import statistics as st
import sys
import time
from pathlib import Path

import torch
from tokenizers import Tokenizer

from relevance_explainer.claims import NO_REPEAT_TAGGED, sanitize_input
from relevance_explainer.tiny.model import Config, TinyLM
from relevance_explainer.tiny.train import EOS, SEP, quantize

torch.set_num_threads(4)
rows = random.Random(0).sample([json.loads(l) for l in open("data/eval_real/itemfirst_nosim_v4/test.jsonl")], 40)
print("| run | median s, one answer | median new tokens | median s, answer + 3 samples in one batch |")
print("|---|---|---|---|")
for run in map(Path, sys.argv[1:]):
    ck = torch.load(run / "model_fp32.pt", map_location="cpu")
    model = TinyLM(Config(**ck["config"]))
    model.load_state_dict(ck["state"])
    model, _ = quantize(model.eval(), 8)
    tk = Tokenizer.from_file(str(run / "tokenizer.json"))
    tagged = tk.token_to_id("<|exp|>") is not None
    nr, max_new = (NO_REPEAT_TAGGED, 420) if tagged else (4, 260)
    one, toks, four = [], [], []
    with torch.no_grad():
        for i, r in enumerate(rows):
            p = tk.encode(sanitize_input(r["input"]) + SEP).ids
            if i == 0:
                model.generate_batch([p], EOS, max_new, no_repeat=nr)  # warm-up
            t = time.time()
            out = model.generate_batch([p], EOS, max_new, no_repeat=nr)[0]
            one.append(time.time() - t)
            toks.append(len(out))
            torch.manual_seed(i)
            t = time.time()
            model.generate_batch([p] * 4, EOS, max_new, no_repeat=nr, temperature=0.7)
            four.append(time.time() - t)
    print(f"| {run.name} | {st.median(one):.2f} | {st.median(toks):.0f} | {st.median(four):.2f} |", flush=True)
