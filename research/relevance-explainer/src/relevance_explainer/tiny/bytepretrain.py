"""Pretrain ByteFlow Net on the byte shards, evaluating bits per byte on the same held-out text as the BPE models.

  CUDA_VISIBLE_DEVICES=1 python -m relevance_explainer.tiny.bytepretrain --root data/corpus12k --out runs/byteflow-a --bytes 2.9e8
"""
from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path

import numpy as np
import torch
from tokenizers import Tokenizer

from .byteflow import BOS, EOS, ByteConfig, ByteFlowNet
from .corpus import LANGS
from .pretrain import HOLDOUT

HOLD_BYTES = 1_000_000  # bytes at the end of each language shard kept out of training


def load_shards(root: Path) -> dict[str, np.memmap]:
    return {l: np.memmap(root / "bytes" / f"{l}.bin", dtype=np.uint8, mode="r") for l in LANGS}


def sampler(shards, alpha: float, seq: int, bs: int, rng: np.random.Generator):
    langs = list(shards)
    sizes = np.array([len(shards[l]) - HOLD_BYTES for l in langs], dtype=float)
    p = sizes**alpha
    p /= p.sum()

    def batch() -> torch.Tensor:
        out = np.empty((bs, seq), dtype=np.int64)
        for i, k in enumerate(rng.choice(len(langs), size=bs, p=p)):
            s = shards[langs[k]]
            o = int(rng.integers(0, len(s) - HOLD_BYTES - seq))
            out[i] = s[o : o + seq]
        x = torch.from_numpy(out)
        x[x == 0] = EOS
        x[:, 0] = BOS
        return x

    return batch, dict(zip(langs, p.round(3)))


def heldout_bytes(root: Path, tk: Tokenizer, seq_tokens: int = 1024, n_seq: int = 64) -> dict[str, np.ndarray]:
    """The exact held-out text of the BPE evaluation (last HOLDOUT tokens, first n_seq * seq_tokens of them), as bytes."""
    out = {}
    for l in LANGS:
        toks = np.memmap(root / "tok" / f"{l}.bin", dtype=np.uint16, mode="r")
        tail = np.asarray(toks[-HOLDOUT:][: n_seq * seq_tokens], dtype=np.int64)
        out[l] = np.frombuffer(tk.decode(tail.tolist()).encode("utf-8"), dtype=np.uint8)
    return out


@torch.no_grad()
def eval_bpb(model: ByteFlowNet, held: dict[str, np.ndarray], seq: int, micro: int) -> dict[str, float]:
    model.eval()
    res = {}
    for l, b in held.items():
        n = (len(b) - 1) // (seq - 1)  # each sequence: BOS + (seq - 1) bytes; every byte after BOS is predicted
        x = torch.full((n, seq), BOS, dtype=torch.long)
        body = torch.from_numpy(np.asarray(b[: n * (seq - 1)], dtype=np.int64)).view(n, seq - 1)
        body[body == 0] = EOS
        x[:, 1:] = body
        tot, cnt = 0.0, 0
        for i in range(0, n, micro):
            xb = x[i : i + micro].cuda()
            with torch.autocast("cuda", dtype=torch.bfloat16):
                _, loss, _ = model(xb, xb)
            tot += loss.item() * xb.numel() - loss.item() * xb.size(0)
            cnt += xb.numel() - xb.size(0)
        res[l] = round(tot / cnt / math.log(2), 4)
    model.train()
    return res


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--bytes", type=float, default=2.9e8)
    ap.add_argument("--seq", type=int, default=3072)
    ap.add_argument("--bs", type=int, default=32)
    ap.add_argument("--micro", type=int, default=8)
    ap.add_argument("--lr", type=float, default=2e-3)
    ap.add_argument("--alpha", type=float, default=0.5)
    ap.add_argument("--rate", default="logdet", choices=["logdet", "l2"])
    ap.add_argument("--ratio", type=float, default=3.0)
    ap.add_argument("--eval-every", type=int, default=500)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()

    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(a.seed)
    root = Path(a.root)
    shards = load_shards(root)
    batch, probs = sampler(shards, a.alpha, a.seq, a.bs, np.random.default_rng(a.seed))
    held = heldout_bytes(root, Tokenizer.from_file(str(root / "tokenizer.json")))
    print("language sampling:", probs, flush=True)
    print("held-out bytes:", {l: len(b) for l, b in held.items()}, flush=True)

    cfg = ByteConfig(rate=a.rate, ratio=a.ratio, max_bytes=max(a.seq, 6144))
    model = ByteFlowNet(cfg).cuda()
    print(f"params {model.n_params() / 1e6:.2f}M (global transformer {sum(p.numel() for p in model.globals.parameters()) / 1e6:.2f}M)", flush=True)
    decay = [p for p in model.parameters() if p.dim() >= 2]
    nodecay = [p for p in model.parameters() if p.dim() < 2]
    opt = torch.optim.AdamW([{"params": decay, "weight_decay": 0.1}, {"params": nodecay, "weight_decay": 0.0}],
                            lr=a.lr, betas=(0.9, 0.95), fused=True)
    steps = int(a.bytes / (a.bs * a.seq))
    warm = min(300, steps // 10)
    lr_at = lambda s: (s + 1) / warm if s < warm else 0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * (s - warm) / max(steps - warm, 1)))
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lr_at)

    log, t0 = [], time.time()
    model.train()
    for step in range(1, steps + 1):
        xb = batch().cuda(non_blocking=True)
        for x in xb.split(a.micro):
            with torch.autocast("cuda", dtype=torch.bfloat16):
                _, loss, _ = model(x, x)
            (loss * len(x) / len(xb)).backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step(), sched.step(), opt.zero_grad(set_to_none=True)
        if step % 50 == 0:
            el = time.time() - t0
            print(f"step {step}/{steps} loss {loss.item():.3f} bpb {loss.item() / math.log(2):.3f} {el / 60:.1f} min "
                  f"{step * a.bs * a.seq / el / 1e3:.0f}k bytes/s", flush=True)
        if step % a.eval_every == 0 or step == steps:
            bpb = eval_bpb(model, held, a.seq, a.micro)
            log.append({"step": step, "bytes": step * a.bs * a.seq, "train_loss": loss.item(), "heldout_bpb": bpb})
            print("heldout bpb:", bpb, "mean", round(sum(bpb.values()) / len(bpb), 4), flush=True)
            torch.save({"config": cfg.to_dict(), "state": model.state_dict()}, out / "model.pt")
            (out / "log.json").write_text(json.dumps({"args": vars(a), "params_m": model.n_params() / 1e6, "log": log}, indent=1))


if __name__ == "__main__":
    main()
