"""Time N explanations written after the same verdict line on CPU (the app's settings: int8 model, 4 threads),
one at a time (as the app's retry does now) and as one batch (generate_batch), for N = 1, 3, 6, 9.

  python scripts/bench_parallel_samples.py [example_id ...]
"""
import re
import sys
import time

import torch

from relevance_explainer import viewer as V
from relevance_explainer.webapp import server as S

torch.set_num_threads(4)
name = "Realistic, no retrieval score"
model, tk, *_ = V.load(name)
ids = sys.argv[1:] or ["a9001-506", "q9001-144", "a9001-31"]
for eid in ids:
    text = re.sub(r"\nRetrieval similarity: [-0-9.]+", "", S.render_for(eid, None)["itemfirst"])
    vline = tk.encode(text + V.SEP + "relevant\n").ids
    model.generate_batch([vline], V.EOS, 260, temperature=0.7)  # warm-up
    row = []
    for n in (1, 3, 6, 9):
        t = time.time()
        for r in range(n):
            torch.manual_seed(r)
            model.generate_batch([vline], V.EOS, 260, temperature=0.7)
        seq = time.time() - t
        torch.manual_seed(0)
        t = time.time()
        outs = model.generate_batch([vline] * n, V.EOS, 260, temperature=0.7)
        par = time.time() - t
        row.append(f"N={n}: one at a time {seq:.2f} s, batch {par:.2f} s (longest {max(map(len, outs))} tokens)")
    print(eid, f"prompt {len(vline)} tokens")
    print("  " + "\n  ".join(row))
