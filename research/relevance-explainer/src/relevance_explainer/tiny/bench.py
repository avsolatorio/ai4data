"""CPU decoding latency of model shapes (weights are random; latency does not depend on their values).

  python -m relevance_explainer.tiny.bench
"""
from __future__ import annotations

import argparse
import time

import torch

from .model import Config, TinyLM

CONFIGS = {
    "8 passes, 8 blocks (no sharing)": dict(n_layer=8, n_unique=0),
    "8 passes, 4 blocks (shared)": dict(n_layer=8, n_unique=4),
    "4 passes, 4 blocks (no sharing)": dict(n_layer=4, n_unique=0),
    "12 passes, 4 blocks (shared)": dict(n_layer=12, n_unique=4),
    "7 passes, 7 blocks (current model)": dict(n_layer=7, n_unique=0),
}


def bench(cfg: Config, prompt_len: int, new_tokens: int, reps: int) -> tuple[float, float]:
    torch.manual_seed(0)
    m = TinyLM(cfg).eval()
    prompt = torch.randint(2, cfg.vocab, (prompt_len,)).tolist()
    m.generate_batch([prompt], -1, 4, no_repeat=0)  # warm-up
    first, per_tok = [], []
    for _ in range(reps):
        t = time.perf_counter()
        m.generate_batch([prompt], -1, 1, no_repeat=0)
        first.append(time.perf_counter() - t)
        t = time.perf_counter()
        m.generate_batch([prompt], -1, new_tokens + 1, no_repeat=0)
        full = time.perf_counter() - t
        per_tok.append((full - first[-1]) / new_tokens)
    return min(first) * 1000, min(per_tok) * 1000


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--threads", type=int, default=1)
    ap.add_argument("--prompt", type=int, default=330)
    ap.add_argument("--new", type=int, default=64)
    ap.add_argument("--reps", type=int, default=5)
    a = ap.parse_args()
    torch.set_num_threads(a.threads)
    base = dict(vocab=12000, d_model=448, n_head=7, d_ff=1024, max_len=1536, dropout=0.0)
    print(f"{a.threads} CPU thread(s), prompt {a.prompt} tokens, fp32 weights")
    print(f"{'configuration':36s} {'params':>8s} {'first token':>12s} {'per token':>10s}")
    for name, extra in CONFIGS.items():
        cfg = Config(**base, **extra)
        first, per = bench(cfg, a.prompt, a.new, a.reps)
        print(f"{name:36s} {TinyLM(cfg).n_params() / 1e6:7.1f}M {first:9.0f} ms {per:7.1f} ms")


if __name__ == "__main__":
    main()
