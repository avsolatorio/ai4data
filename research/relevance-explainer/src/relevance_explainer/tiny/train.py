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
from .scriptguard import banned_for, token_masks

PAD, EOS = 0, 1


def train_tokenizer(rows: list[dict], vocab: int, path: Path) -> Tokenizer:
    tk = ByteLevelBPETokenizer()
    tk.train_from_iterator((t for r in rows for t in (r["input"], r["output"])), vocab_size=vocab,
                           special_tokens=["<pad>", "<eos>"], show_progress=False)
    tk.save(str(path))
    return Tokenizer.from_file(str(path))


VERDICT_TAG = "\nVerdict:"


def from_verdict_last(text: str) -> str:
    """'narrative\\nVerdict: x' -> 'x\\nnarrative', the format the parsers and judges expect."""
    narrative, tag, verdict = text.rpartition(VERDICT_TAG)
    return f"{verdict.strip()}\n{narrative.strip()}" if tag else f"invalid\n{text.strip()}"


def encode(tk: Tokenizer, rows: list[dict], max_len: int, verdict_last: bool = False) -> tuple[list[dict], int]:
    """Prompt and target ids. With `verdict_last` the target is the narrative followed by 'Verdict: <verdict>';
    `vstart` marks where the verdict tokens begin, so that they get the verdict loss weight."""
    out, dropped = [], 0
    for idx, r in enumerate(rows):
        p = tk.encode(r["input"] + SEP).ids
        if verdict_last:
            head = tk.encode(r["output"].split("\n", 1)[1].strip() + VERDICT_TAG).ids
            vt = tk.encode(" " + r["verdict"]).ids
            t, vstart, v = head + vt + [EOS], len(p) + len(head), len(vt)
        else:
            t = tk.encode(r["output"]).ids + [EOS]
            v = len(tk.encode(r["verdict"]).ids) if "verdict" in r else 1  # tokens that carry the verdict word
            vstart = len(p)
        if len(p) + len(t) > max_len:
            dropped += 1
            continue
        out.append({"ids": p + t, "labels": [-100] * len(p) + t, "plen": len(p), "vstart": vstart, "vlen": v, "idx": idx, "row": r})
    return out, dropped


def batches(data, bs, shuffle, rng, vw: float = 1.0, soft: dict | None = None, pw: float = 0.0):
    """Yield (ids, labels, weights, soft_rows). Verdict tokens get loss weight `vw`, narrative tokens 1.

    `soft` maps a row index to the teacher's verdict probabilities. For such rows the first verdict token is left out
    of the weighted cross-entropy and returned in `soft_rows` as (batch row, prompt length, probabilities)."""
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
        soft_rows = []
        for k, i in enumerate(ch):
            L = len(data[i]["ids"])
            ids[k, :L], lab[k, :L] = torch.tensor(data[i]["ids"]), torch.tensor(data[i]["labels"])
            vs = data[i].get("vstart", data[i]["plen"])
            wt[k, vs : vs + data[i]["vlen"]] = vw
            if pw:  # language-model loss on the input tokens too, with weight pw (position 0 has no prediction)
                p = data[i]["plen"]
                lab[k, 1:p] = ids[k, 1:p]
                wt[k, :p] = pw
            if soft and data[i]["idx"] in soft:
                wt[k, data[i]["plen"]] = 0.0
                soft_rows.append((k, data[i]["plen"], soft[data[i]["idx"]]))
        yield ids.cuda(), lab.cuda(), wt.cuda(), soft_rows


@torch.no_grad()
def val_loss(model, data, bs) -> float:
    model.eval()
    tot, n = 0.0, 0
    for ids, lab, _, _ in batches(data, bs, False, random.Random(0)):
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
            logits = model(ids.cuda(), prefix=torch.tensor([data[i]["plen"] for i in ch]).cuda() if model.c.prefix_lm else None)
        for k, i in enumerate(ch):  # with the verdict last this is teacher-forced on the reference narrative
            p = data[i].get("vstart", data[i]["plen"])
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
def evaluate(model: TinyLM, tk: Tokenizer, enc: list[dict], max_new: int, batch: int = 48,
             copy_bias: float = 0.0, script_guard: bool = False) -> tuple[dict, list[dict]]:
    model.eval()
    prompts = [tk.encode(e["row"]["input"] + SEP).ids for e in enc]
    order = sorted(range(len(prompts)), key=lambda i: len(prompts[i]))  # similar lengths per batch limits padding
    gens = [""] * len(prompts)
    masks = token_masks(tk) if script_guard else None
    for s in range(0, len(order), batch):
        ch = order[s : s + batch]
        ban = torch.stack([banned_for(masks, enc[i]["row"]["input"]) for i in ch]) if masks else None
        with torch.autocast("cuda", dtype=torch.bfloat16):
            outs = model.generate_batch([prompts[i] for i in ch], EOS, max_new, copy_bias=copy_bias, banned=ban)
        for i, o in zip(ch, outs):
            gens[i] = tk.decode(o).strip()
            if model.c.verdict_last:
                gens[i] = from_verdict_last(gens[i])
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
    ap.add_argument("--soft-weight", type=float, default=1.0, help="weight of the soft verdict cross-entropy")
    ap.add_argument("--soft", action="store_true", help="use teacher verdict probabilities (soft_train.jsonl) for the first verdict token")
    ap.add_argument("--select", choices=["val_loss", "val_verdict_acc"], default="val_loss", help="how to pick the best epoch")
    ap.add_argument("--lora-rank", type=int, default=0, help="train a low-rank adapter of this rank instead of all weights")
    ap.add_argument("--lora-alpha", type=float, default=32.0)
    ap.add_argument("--tokenizer", default=None, help="prebuilt tokenizer.json; trained on the task data when omitted")
    ap.add_argument("--init", default=None, help="pretrained model.pt (config and weights)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--copy", action="store_true", help="add a pointer (copy) head to the initialized model")
    ap.add_argument("--prefix-lm", action="store_true", help="bidirectional attention within the input (prefix LM)")
    ap.add_argument("--verdict-last", action="store_true", help="target is the narrative followed by 'Verdict: <verdict>'")
    ap.add_argument("--prompt-loss", type=float, default=0.0, help="loss weight on the input tokens (0: output only)")
    a = ap.parse_args()
    assert not (a.prompt_loss and a.prefix_lm), "the prefix LM takes the prompt length from the labels"

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
        enc[s], dropped = encode(tk, rows, a.max_len, a.verdict_last)
        info[s] = {"examples": len(rows), "dropped_too_long": dropped}
    lens = sorted(len(x["ids"]) for x in enc["train"])
    info["train_tokens_median_p95_max"] = [lens[len(lens) // 2], lens[int(len(lens) * 0.95)], lens[-1]]
    print("data:", info, flush=True)
    first_ids = [tk.encode(c).ids[0] for c in VERDICTS]  # first token of each verdict word, in VERDICTS order
    assert len(set(first_ids)) == 3, "verdict words must start with distinct tokens"
    soft = None
    if a.soft:
        soft = {d_["i"]: d_["p"] for d_ in map(json.loads, (d / "soft_train.jsonl").read_text().splitlines())}
        info["soft_labels"] = len(soft)
        print(f"soft verdict labels for {len(soft)} training examples", flush=True)

    if a.init:
        ck = torch.load(a.init, map_location="cpu")
        cfg = Config(**{**ck["config"], "dropout": a.dropout, **({"copy": True} if a.copy else {}),
                        **({"prefix_lm": True} if a.prefix_lm else {}), **({"verdict_last": True} if a.verdict_last else {})})
        model = TinyLM(cfg)
        missing, unexpected = model.load_state_dict(ck["state"], strict=False)
        assert not unexpected and all(k.startswith("copy_") for k in missing), (missing, unexpected)
        model = model.cuda()
        a.max_len = min(a.max_len, cfg.max_len)
    else:
        cfg = Config(vocab=tk.get_vocab_size(), d_model=a.d_model, n_layer=a.layers, n_head=a.heads, d_ff=a.d_ff,
                     max_len=a.max_len, dropout=a.dropout)
        model = TinyLM(cfg).cuda()
    print(f"params: {model.n_params() / 1e6:.2f}M", flush=True)
    if a.lora_rank:
        from .lora import add_lora, lora_bytes, lora_state, merge_lora

        add_lora(model, a.lora_rank, a.lora_alpha, a.dropout)
        print(f"lora rank {a.lora_rank}: {sum(p.numel() for p in model.parameters() if p.requires_grad) / 1e6:.2f}M trainable parameters", flush=True)
    decay = [p for p in model.parameters() if p.requires_grad and p.dim() >= 2]
    nodecay = [p for p in model.parameters() if p.requires_grad and p.dim() < 2]
    opt = torch.optim.AdamW([{"params": decay, "weight_decay": 0.1}, {"params": nodecay, "weight_decay": 0.0}], lr=a.lr, betas=(0.9, 0.95))
    steps = a.epochs * math.ceil(len(enc["train"]) / a.bs)
    warm = min(100, steps // 10)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: (s + 1) / warm if s < warm else 0.5 * (1 + math.cos(math.pi * (s - warm) / max(steps - warm, 1))) * 0.95 + 0.05)

    log, best, best_state, t0 = [], 1e9, None, time.time()
    for ep in range(a.epochs):
        model.train()
        run, k = 0.0, 0
        for ids, lab, wt, soft_rows in batches(enc["train"], a.bs, True, rng, a.verdict_weight, soft, a.prompt_loss):
            with torch.autocast("cuda", dtype=torch.bfloat16):
                logits, loss = model(ids, lab, wt)
                if soft_rows:  # cross-entropy of the first verdict token against the teacher's probabilities
                    b, pl, pr = (torch.tensor(x) for x in zip(*soft_rows))
                    lp = torch.log_softmax(logits[b.cuda(), (pl - 1).cuda()].float(), dim=-1)[:, first_ids]
                    loss = loss + a.soft_weight * -(pr.cuda().float() * lp).sum(-1).mean()
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
    if a.lora_rank:  # keep the adapter on its own, then fold it into the base for evaluation and export
        adapter = lora_state(model)
        torch.save({"rank": a.lora_rank, "alpha": a.lora_alpha, "state": adapter}, out / "adapter.pt")
        info["adapter_bytes_int8"] = lora_bytes(adapter, 8)
        merge_lora(model)
        best_state = model.state_dict()
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
