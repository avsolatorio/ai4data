"""Task fine-tuning and evaluation of ByteFlow Net on bytes: (input text) -> (verdict line + narrative).

The prompt is BOS + input bytes + separator; the target is the output bytes + EOS. Loss on the target only, with a
higher weight on the verdict bytes, as for the BPE model. Evaluation generates for a fixed seeded subset of the test
set (the same 800 examples the judge uses) and reports verdict accuracy on that subset.

  CUDA_VISIBLE_DEVICES=0 python -m relevance_explainer.tiny.bytetrain --data data/f2 --out runs/byteflow-task \
      --init runs/byteflow-a/model.pt --epochs 6
"""
from __future__ import annotations

import argparse
import json
import math
import random
import time
from collections import Counter
from pathlib import Path

import torch

from ..train.finetune import VERDICTS, macro_f1, parse, read
from .byteflow import BOS, EOS, ByteConfig, ByteFlowNet

SEP = b"\n\n### Judgment\n"


def encode(rows: list[dict], max_len: int) -> tuple[list[dict], int]:
    out, dropped = [], 0
    for r in rows:
        p = [BOS] + list(r["input"].encode("utf-8") + SEP)
        t = list(r["output"].encode("utf-8")) + [EOS]
        if len(p) + len(t) > max_len:
            dropped += 1
            continue
        vlen = len(r["verdict"].encode("utf-8"))
        out.append({"ids": p + t, "plen": len(p), "vlen": vlen, "row": r})
    return out, dropped


def batches(data, bs, shuffle, rng, vw: float):
    idx = list(range(len(data)))
    if shuffle:
        rng.shuffle(idx)
        idx = [i for c in range(0, len(idx), bs * 8) for i in sorted(idx[c : c + bs * 8], key=lambda i: len(data[i]["ids"]))]
    chunks = [idx[i : i + bs] for i in range(0, len(idx), bs)]
    if shuffle:
        rng.shuffle(chunks)
    for ch in chunks:
        n = max(len(data[i]["ids"]) for i in ch)
        ids = torch.full((len(ch), n), EOS, dtype=torch.long)  # padding byte after EOS; masked out of the loss
        lab = torch.full((len(ch), n), -100, dtype=torch.long)
        wt = torch.ones((len(ch), n))
        valid = torch.zeros((len(ch), n), dtype=torch.bool)
        for k, i in enumerate(ch):
            d = data[i]
            L = len(d["ids"])
            ids[k, :L] = torch.tensor(d["ids"])
            lab[k, d["plen"] : L] = ids[k, d["plen"] : L]
            wt[k, d["plen"] : d["plen"] + d["vlen"]] = vw
            valid[k, :L] = True
        yield ids.cuda(), lab.cuda(), wt.cuda(), valid.cuda()


def weighted_loss(model, ids, lab, wt, valid):
    logits, _ = model(ids, valid=valid)
    flat = logits[:, :-1].reshape(-1, logits.size(-1)).float()
    target = lab[:, 1:].reshape(-1)
    ce = torch.nn.functional.cross_entropy(flat, target, ignore_index=-100, reduction="none")
    w = wt[:, 1:].reshape(-1) * (target != -100)
    return (ce * w).sum() / w.sum().clamp_min(1e-8)


@torch.no_grad()
def val_loss(model, data, bs) -> float:
    model.eval()
    tot, n = 0.0, 0
    for ids, lab, wt, valid in batches(data, bs, False, random.Random(0), 1.0):
        with torch.autocast("cuda", dtype=torch.bfloat16):
            loss = weighted_loss(model, ids, lab, torch.ones_like(wt), valid)
        k = (lab[:, 1:] != -100).sum().item()
        tot, n = tot + loss.item() * k, n + k
    model.train()
    return tot / n


@torch.no_grad()
def generate_subset(model, rows: list[dict], max_new: int) -> list[str]:
    model.eval()
    outs = []
    for r in rows:
        p = [BOS] + list(r["input"].encode("utf-8") + SEP)
        if len(p) >= model.c.max_bytes - max_new:
            outs.append("")
            continue
        with torch.autocast("cuda", dtype=torch.bfloat16):
            b = model.generate(p, max_new=max_new)
        outs.append(bytes(x for x in b if x < 256).decode("utf-8", errors="replace").strip())
    return outs


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--init", required=True)
    ap.add_argument("--epochs", type=int, default=6)
    ap.add_argument("--lr", type=float, default=5e-4)
    ap.add_argument("--bs", type=int, default=16)
    ap.add_argument("--micro", type=int, default=8)
    ap.add_argument("--max-len", type=int, default=6144)
    ap.add_argument("--max-new", type=int, default=400)
    ap.add_argument("--verdict-weight", type=float, default=5.0)
    ap.add_argument("--eval-n", type=int, default=800)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()

    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(a.seed)
    rng = random.Random(a.seed)
    d = Path(a.data)
    raw = {s: read(d / f"{s}.jsonl") for s in ("train", "val", "test")}
    enc, info = {}, {}
    for s, rows in raw.items():
        enc[s], dropped = encode(rows, a.max_len)
        info[s] = {"examples": len(rows), "dropped_too_long": dropped}
    lens = sorted(len(x["ids"]) for x in enc["train"])
    info["train_bytes_median_p95_max"] = [lens[len(lens) // 2], lens[int(len(lens) * 0.95)], lens[-1]]
    print("data:", info, flush=True)

    ck = torch.load(a.init, map_location="cpu")
    cfg = ByteConfig(**{**ck["config"], "max_bytes": max(ck["config"]["max_bytes"], a.max_len)})
    model = ByteFlowNet(cfg)
    model.load_state_dict(ck["state"], strict=False)
    model = model.cuda()
    decay = [p for p in model.parameters() if p.dim() >= 2]
    nodecay = [p for p in model.parameters() if p.dim() < 2]
    opt = torch.optim.AdamW([{"params": decay, "weight_decay": 0.1}, {"params": nodecay, "weight_decay": 0.0}], lr=a.lr, betas=(0.9, 0.95))
    steps = a.epochs * math.ceil(len(enc["train"]) / a.bs)
    warm = min(100, steps // 10)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: (s + 1) / warm if s < warm else 0.5 * (1 + math.cos(math.pi * (s - warm) / max(steps - warm, 1))) * 0.95 + 0.05)

    log, best, best_state, t0 = [], 1e9, None, time.time()
    for ep in range(a.epochs):
        model.train()
        run, k = 0.0, 0
        for ids, lab, wt, valid in batches(enc["train"], a.bs, True, rng, a.verdict_weight):
            for sl in range(0, ids.size(0), a.micro):
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    loss = weighted_loss(model, ids[sl : sl + a.micro], lab[sl : sl + a.micro], wt[sl : sl + a.micro], valid[sl : sl + a.micro])
                (loss * min(a.micro, ids.size(0) - sl) / ids.size(0)).backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step(), sched.step(), opt.zero_grad(set_to_none=True)
            run, k = run + loss.item(), k + 1
        vl = val_loss(model, enc["val"], a.bs)
        log.append({"epoch": ep + 1, "train_loss": run / k, "val_loss": vl, "minutes": (time.time() - t0) / 60})
        if vl < best:
            best, best_state = vl, {k_: v.detach().cpu().clone() for k_, v in model.state_dict().items()}
            log[-1]["best"] = True
        print(log[-1], flush=True)
    model.load_state_dict(best_state)
    torch.save({"config": cfg.to_dict(), "state": best_state}, out / "model_fp32.pt")

    # generation on the judge's subset: the same seeded sample of the test set
    sub = random.Random(0).sample(raw["test"], min(a.eval_n, len(raw["test"])))
    t1 = time.time()
    gens = generate_subset(model, sub, a.max_new)
    pred = [parse(g) for g in gens]
    gold = [r["verdict"] for r in sub]
    pv = [p[0] for p in pred]
    metrics = {"params_million": round(model.n_params() / 1e6, 2), "data": info, "train_log": log, "best_val_loss": best,
               "eval_n": len(sub), "verdict_accuracy": sum(g == p for g, p in zip(gold, pv)) / len(gold),
               "majority_baseline": Counter(gold).most_common(1)[0][1] / len(gold), "f1": macro_f1(gold, pv),
               "invalid_verdict_rate": pv.count("invalid") / len(pv), "gen_seconds_per_example": (time.time() - t1) / len(sub),
               "narrative_words_median": sorted(len(p[1].split()) for p in pred)[len(pred) // 2]}
    (out / "metrics.json").write_text(json.dumps(metrics, indent=2, ensure_ascii=False))
    with (out / "test_generations_fp32.jsonl").open("w") as f:
        for r, g in zip(sub, gens):
            f.write(json.dumps({"input": r["input"], "reference": r["output"], "generated": g, "language": r["language"], "source": r["source"]}, ensure_ascii=False) + "\n")
    print(json.dumps({k: metrics[k] for k in ("verdict_accuracy", "majority_baseline", "f1", "invalid_verdict_rate", "gen_seconds_per_example", "narrative_words_median")}, indent=2))


if __name__ == "__main__":
    main()
