"""Time of the small occupation model (arm all, seed 42) per survey-style answer.  python speed.py cpu|cuda [THREADS]

One answer at a time (200 answers after 10 for warm-up) and in batches of 64 (all 1,000). Three repeats; the
median of the repeats is reported."""
import json, sys, time

import numpy as np
import pandas as pd
import torch

from common import OUT, TASK, read_jsonl
from ai4data.small_ai import model as mdl

device = sys.argv[1]
threads = int(sys.argv[2]) if len(sys.argv) > 2 else 4
if device == 'cpu':
    torch.set_num_threads(threads)
docs = pd.DataFrame(read_jsonl(OUT / 'test_survey/examples.jsonl'))
texts = docs['text'].tolist()
config, tokenizer, model, device = mdl.load(OUT / 'arms/all/seed42/model', device)
n_params = sum(p.numel() for p in model.parameters())
sync = (lambda: torch.cuda.synchronize()) if device == 'cuda' else (lambda: None)
one, many = [], []
for rep in range(3):
    for t in texts[:10]:
        mdl.probabilities([t], config, tokenizer, model, device, 64)
    sync(); a = time.perf_counter()
    for t in texts[10:210]:
        mdl.probabilities([t], config, tokenizer, model, device, 64)
    sync(); one.append((time.perf_counter() - a) / 200)
    a = time.perf_counter()
    p = mdl.probabilities(texts, config, tokenizer, model, device, 64)
    sync(); many.append((time.perf_counter() - a) / len(texts))
order = np.argsort(-p, axis=1)[:, :5]
labels = config['labels']
gold = docs['labels'].map(lambda x: x[0]).tolist()
top = {k: float(np.mean([g in [labels[j] for j in row[:k]] for g, row in zip(gold, order)])) for k in (1, 3, 5)}
out = {'device': device, 'threads': threads if device == 'cpu' else None, 'parameters': n_params,
       'one_at_a_time_seconds': [round(x, 5) for x in one], 'one_at_a_time_median': round(float(np.median(one)), 5),
       'batched_seconds': [round(x, 6) for x in many], 'batched_median': round(float(np.median(many)), 6),
       'top': top, 'torch_threads': torch.get_num_threads()}
(OUT / 'direct').mkdir(exist_ok=True)
(OUT / 'direct' / f'speed_{device}{threads if device == "cpu" else ""}.json').write_text(json.dumps(out, indent=1))
print(json.dumps(out))
