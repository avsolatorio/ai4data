"""Train the ~20 MB model from scratch on the assembled splits, then evaluate fp32, INT8 and INT4 weights.

Example:
  CUDA_VISIBLE_DEVICES=0 python -m relevance_explainer.tiny.train --data data/v1 --out runs/tiny-20m
"""
from __future__ import annotations

import argparse
import copy
import json
import math
import random
import time
from collections import Counter
from pathlib import Path

import torch
from tokenizers import ByteLevelBPETokenizer, Tokenizer

from ..train.finetune import SCRIPTS, SEP, VERDICTS, macro_f1, parse, read, script_share
from .model import Config, TinyLM

PAD, EOS = 0, 1


def train_tokenizer(rows: list[dict], vocab: int, path: Path) -> Tokenizer:
    tk = ByteLevelBPETokenizer()
    tk.train_from_iterator((t for r in rows for t in (r["input"], r["output"])), vocab_size=vocab,
                           special_tokens=["<pad>", "<eos>"], show_progress=False)
    tk.save(str(path))
    return Tokenizer.from_file(str(path))


def encode(tk: Tokenizer, rows: list[dict], max_len: int) -> tuple[list[dict], int]:
    out, dropped = [], 0
    for r in rows:
        p = tk.encode(r["input"] + SEP).ids
        t = tk.encode(r["output"]).ids + [EOS]
        if len(p) + len(t) > max_len:
            dropped += 1
            continue
        v = len(tk.encode(r["verdict"]).ids) if "verdict" in r else 1  # tokens that carry the verdict word
        out.append({"ids": p + t, "labels": [-100] * len(p) + t, "plen": len(p), "vlen": v, "row": r})
    return out, dropped


def batches(data, bs, shuffle, rng, vw: float = 1.0):
    """Yield (ids, labels, weights). Verdict tokens get loss weight `vw`, narrative tokens 1."""
    idx = list(range(len(data)))
    if shuffle:
        rng.shuffle(idx)
        idx = [i for c in range(0, len(idx), bs * 8) for i in sorted(idx[c : c + bs * 8], key=lambda i: len(data[i]["ids"]))]
    chunks = [idx[i : i + bs] for i in range(0, len(idx), bs)]
    if shuffle:
        rng.shuffle(chunks)
    for ch in chunks:
        n = max(len(data[i]["ids"]) for i in ch)
        ids, lab = torch.full((len(ch), n), PAD), torch.full((len(ch), n), -100)
        wt = torch.ones((len(ch), n))
        for k, i in enumerate(ch):
            L = len(data[i]["ids"])
            ids[k, :L], lab[k, :L] = torch.tensor(data[i]["ids"]), torch.tensor(data[i]["labels"])
            wt[k, data[i]["plen"] : data[i]["plen"] + data[i]["vlen"]] = vw
        yield ids.cuda(), lab.cuda(), wt.cuda()


@torch.no_grad()
def val_loss(model, data, bs) -> float:
    model.eval()
    tot, n = 0.0, 0
    for ids, lab, _ in batches(data, bs, False, random.Random(0)):
        with torch.autocast("cuda", dtype=torch.bfloat16):
            _, loss = model(ids, lab)
        k = (lab[:, 1:] != -100).sum().item()
        tot, n = tot + loss.item() * k, n + k
    return tot / n


@torch.no_grad()
def val_verdict_acc(model, data, bs) -> float:
    """Teacher-forced check: does the first verdict token get the largest logit? Fast proxy for verdict accuracy."""
    model.eval()
    ok = 0
    order = list(range(len(data)))
    for s in range(0, len(order), bs):
        ch = order[s : s + bs]
        n = max(len(data[i]["ids"]) for i in ch)
        ids = torch.full((len(ch), n), PAD)
        for k, i in enumerate(ch):
            ids[k, : len(data[i]["ids"])] = torch.tensor(data[i]["ids"])
        with torch.autocast("cuda", dtype=torch.bfloat16):
            logits = model(ids.cuda())
        for k, i in enumerate(ch):
            p = data[i]["plen"]
            ok += int(logits[k, p - 1].argmax().item() == data[i]["ids"][p])
    return ok / len(order)


def quantize(model: TinyLM, bits: int) -> tuple[TinyLM, int]:
    """Symmetric per-row weight quantization of every 2-D tensor. Returns a dequantized copy and the packed size in bytes."""
    q = copy.deepcopy(model)
    levels, size = 2 ** (bits - 1) - 1, 0
    for name, p in q.named_parameters():
        if p.dim() == 2:
            scale = p.detach().abs().amax(dim=1, keepdim=True).clamp_min(1e-8) / levels
            p.data = (p.detach() / scale).round().clamp(-levels, levels) * scale
            size += math.ceil(p.numel() * bits / 8) + scale.numel() * 2  # fp16 row scales
        else:
            size += p.numel() * 2  # fp16 norms
    return q, size


@torch.no_grad()
def evaluate(model: TinyLM, tk: Tokenizer, enc: list[dict], max_new: int, batch: int = 48) -> tuple[dict, list[dict]]:
    model.eval()
    prompts = [tk.encode(e["row"]["input"] + SEP).ids for e in enc]
    order = sorted(range(len(prompts)), key=lambda i: len(prompts[i]))  # similar lengths per batch limits padding
    gens = [""] * len(prompts)
    for s in range(0, len(order), batch):
        ch = order[s : s + batch]
        with torch.autocast("cuda", dtype=torch.bfloat16):
            outs = model.generate_batch([prompts[i] for i in ch], EOS, max_new)
        for i, o in zip(ch, outs):
            gens[i] = tk.decode(o).strip()
    rows = [e["row"] for e in enc]
    pred = [parse(g) for g in gens]
    gold, pv = [r["verdict"] for r in rows], [p[0] for p in pred]
    maj = Counter(gold).most_common(1)[0]
    by = {}
    for key in ("source", "language"):
        for val in sorted({r[key] for r in rows}):
            idx = [i for i, r in enumerate(rows) if r[key] == val]
            by[f"{key}={val}"] = {"n": len(idx), "acc": sum(gold[i] == pv[i] for i in idx) / len(idx)}
    script = {}
    for lang, name in SCRIPTS.items():
        idx = [i for i, r in enumerate(rows) if r["language"] == lang and pred[i][1]]
        if idx:
            script[lang] = {"n": len(idx), "script_share": sum(script_share(pred[i][1], name) for i in idx) / len(idx)}
    m = {"test_n": len(rows), "verdict_accuracy": sum(g == p for g, p in zip(gold, pv)) / len(gold),
         "majority_baseline": {"class": maj[0], "accuracy": maj[1] / len(gold)}, "f1": macro_f1(gold, pv),
         "invalid_verdict_rate": pv.count("invalid") / len(pv),
         "confusion": {f"{g}->{p}": c for (g, p), c in sorted(Counter(zip(gold, pv)).items())},
         "by_group": by, "script_match": script,
         "narrative_words_median": sorted(len(p[1].split()) for p in pred)[len(pred) // 2]}
    return m, [{"input": r["input"], "reference": r["output"], "generated": g, "language": r["language"], "source": r["source"]}
               for r, g in zip(rows, gens)]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--vocab", type=int, default=8000)
    ap.add_argument("--d-model", type=int, default=384)
    ap.add_argument("--layers", type=int, default=10)
    ap.add_argument("--heads", type=int, default=6)
    ap.add_argument("--d-ff", type=int, default=1024)
    ap.add_argument("--dropout", type=float, default=0.1)
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--bs", type=int, default=16)
    ap.add_argument("--max-len", type=int, default=1280)
    ap.add_argument("--max-new", type=int, default=260)
    ap.add_argument("--train-file", default="train.jsonl")
    ap.add_argument("--verdict-weight", type=float, default=1.0, help="loss weight on the verdict tokens")
    ap.add_argument("--select", choices=["val_loss", "val_verdict_acc"], default="val_loss", help="how to pick the best epoch")
    ap.add_argument("--tokenizer", default=None, help="prebuilt tokenizer.json; trained on the task data when omitted")
    ap.add_argument("--init", default=None, help="pretrained model.pt (config and weights)")
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()

    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    random.seed(a.seed)
    torch.manual_seed(a.seed)
    rng = random.Random(a.seed)
    d = Path(a.data)
    raw = {"train": read(d / a.train_file), "val": read(d / "val.jsonl"), "test": read(d / "test.jsonl")}

    if a.tokenizer:
        tk = Tokenizer.from_file(a.tokenizer)
        tk.save(str(out / "tokenizer.json"))
    else:
        tk = train_tokenizer(raw["train"] + raw["val"], a.vocab, out / "tokenizer.json")
    enc, info = {}, {}
    for s, rows in raw.items():
        enc[s], dropped = encode(tk, rows, a.max_len)
        info[s] = {"examples": len(rows), "dropped_too_long": dropped}
    lens = sorted(len(x["ids"]) for x in enc["train"])
    info["train_tokens_median_p95_max"] = [lens[len(lens) // 2], lens[int(len(lens) * 0.95)], lens[-1]]
    print("data:", info, flush=True)

    if a.init:
        ck = torch.load(a.init, map_location="cpu")
        cfg = Config(**{**ck["config"], "dropout": a.dropout})
        model = TinyLM(cfg)
        model.load_state_dict(ck["state"])
        model = model.cuda()
        a.max_len = min(a.max_len, cfg.max_len)
    else:
        cfg = Config(vocab=tk.get_vocab_size(), d_model=a.d_model, n_layer=a.layers, n_head=a.heads, d_ff=a.d_ff,
                     max_len=a.max_len, dropout=a.dropout)
        model = TinyLM(cfg).cuda()
    print(f"params: {model.n_params() / 1e6:.2f}M", flush=True)
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
        for ids, lab, wt in batches(enc["train"], a.bs, True, rng, a.verdict_weight):
            with torch.autocast("cuda", dtype=torch.bfloat16):
                _, loss = model(ids, lab, wt)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step(), sched.step(), opt.zero_grad(set_to_none=True)
            run, k = run + loss.item(), k + 1
        vl = val_loss(model, enc["val"], a.bs)
        log.append({"epoch": ep + 1, "train_loss": run / k, "val_loss": vl,
                    "val_verdict_acc": val_verdict_acc(model, enc["val"], a.bs)})
        score = vl if a.select == "val_loss" else -log[-1]["val_verdict_acc"]
        if score < best:
            best, best_state = score, copy.deepcopy(model.state_dict())
            log[-1]["best"] = True
        print(log[-1], f"{(time.time() - t0) / 60:.1f} min", flush=True)
    model.load_state_dict(best_state)
    best_ep = max((l["epoch"] for l in log if l.get("best")), default=a.epochs)
    print(f"best val_loss {best:.4f} at epoch {best_ep}", flush=True)
    torch.save({"config": cfg.to_dict(), "state": best_state}, out / "model_fp32.pt")

    results = {"params_million": round(model.n_params() / 1e6, 2), "config": cfg.to_dict(), "data": info,
               "best_epoch": best_ep, "best_val_loss": best, "train_log": log, "precision": {}}
    for name, bits in (("fp32", None), ("int8", 8), ("int4", 4)):
        m, size = (model, model.n_params() * 4) if bits is None else quantize(model, bits)
        metrics, gens = evaluate(m, tk, enc["test"], a.max_new)
        metrics["size_mb"] = round(size / 1e6, 2)
        results["precision"][name] = metrics
        with (out / f"test_generations_{name}.jsonl").open("w") as f:
            for g in gens:
                f.write(json.dumps(g, ensure_ascii=False) + "\n")
        print(name, {k: metrics[k] for k in ("size_mb", "verdict_accuracy", "invalid_verdict_rate", "narrative_words_median")}, flush=True)
        (out / "metrics.json").write_text(json.dumps(results, indent=2, ensure_ascii=False))
    print(json.dumps({k: results["precision"]["int8"][k] for k in ("f1", "confusion", "script_match")}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
