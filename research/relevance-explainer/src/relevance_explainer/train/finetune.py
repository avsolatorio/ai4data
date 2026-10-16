"""Full fine-tune of a small causal LM on (input -> verdict + narrative) and evaluate on the test split.

Example:
  CUDA_VISIBLE_DEVICES=0 python -m relevance_explainer.train.finetune \
      --data data/v1 --model Qwen/Qwen3-0.6B --out runs/qwen3-0.6b-smoke
"""
from __future__ import annotations

import argparse
import json
import math
import random
import time
import unicodedata
from collections import Counter
from pathlib import Path

import torch

VERDICTS = ["relevant", "partly_relevant", "irrelevant"]
SEP = "\n\n### Judgment\n"
SCRIPTS = {"Arabic": "ARABIC", "Hindi": "DEVANAGARI", "Chinese": "CJK", "Russian": "CYRILLIC"}


def read(p: Path) -> list[dict]:
    return [json.loads(l) for l in p.read_text().splitlines()]


def encode(tok, rows: list[dict], max_len: int) -> tuple[list[dict], int]:
    out, dropped = [], 0
    for r in rows:
        prompt = tok(r["input"] + SEP, add_special_tokens=False)["input_ids"]
        target = tok(r["output"] + tok.eos_token, add_special_tokens=False)["input_ids"]
        if len(prompt) + len(target) > max_len:
            dropped += 1
            continue
        out.append({"ids": prompt + target, "labels": [-100] * len(prompt) + target})
    return out, dropped


def batches(data: list[dict], bs: int, pad_id: int, shuffle: bool, rng: random.Random):
    idx = list(range(len(data)))
    if shuffle:
        rng.shuffle(idx)
        # bucket by length inside chunks of 8 batches to limit padding
        idx = [i for c in range(0, len(idx), bs * 8) for i in sorted(idx[c : c + bs * 8], key=lambda i: len(data[i]["ids"]))]
    chunks = [idx[i : i + bs] for i in range(0, len(idx), bs)]
    if shuffle:
        rng.shuffle(chunks)
    for ch in chunks:
        n = max(len(data[i]["ids"]) for i in ch)
        ids = torch.full((len(ch), n), pad_id)
        lab = torch.full((len(ch), n), -100)
        att = torch.zeros((len(ch), n), dtype=torch.long)
        for k, i in enumerate(ch):
            L = len(data[i]["ids"])
            ids[k, :L], lab[k, :L], att[k, :L] = torch.tensor(data[i]["ids"]), torch.tensor(data[i]["labels"]), 1
        yield ids.cuda(), att.cuda(), lab.cuda()


@torch.no_grad()
def eval_loss(model, data, bs, pad_id) -> float:
    model.eval()
    tot, n = 0.0, 0
    for ids, att, lab in batches(data, bs, pad_id, False, random.Random(0)):
        with torch.autocast("cuda", dtype=torch.bfloat16):
            out = model(input_ids=ids, attention_mask=att, labels=lab)
        k = (lab[:, 1:] != -100).sum().item()
        tot, n = tot + out.loss.item() * k, n + k
    return tot / max(n, 1)


@torch.no_grad()
def generate(model, tok, rows: list[dict], bs: int, max_new: int) -> list[str]:
    model.eval()
    tok.padding_side = "left"
    outs: list[str] = []
    order = sorted(range(len(rows)), key=lambda i: len(rows[i]["input"]))
    res = {}
    for s in range(0, len(order), bs):
        ch = order[s : s + bs]
        enc = tok([rows[i]["input"] + SEP for i in ch], return_tensors="pt", padding=True, add_special_tokens=False).to("cuda")
        with torch.autocast("cuda", dtype=torch.bfloat16):
            g = model.generate(**enc, max_new_tokens=max_new, do_sample=False, eos_token_id=tok.eos_token_id, pad_token_id=tok.pad_token_id)
        for i, seq in zip(ch, g):
            res[i] = tok.decode(seq[enc["input_ids"].shape[1] :], skip_special_tokens=True).strip()
    return [res[i] for i in range(len(rows))]


def parse(text: str) -> tuple[str, str]:
    head, _, rest = text.partition("\n")
    v = head.strip().lower()
    return (v if v in VERDICTS else "invalid"), rest.strip()


def macro_f1(gold: list[str], pred: list[str]) -> dict:
    f1 = {}
    for c in VERDICTS:
        tp = sum(g == c and p == c for g, p in zip(gold, pred))
        fp = sum(g != c and p == c for g, p in zip(gold, pred))
        fn = sum(g == c and p != c for g, p in zip(gold, pred))
        f1[c] = 2 * tp / max(2 * tp + fp + fn, 1)
    return {"per_class": f1, "macro": sum(f1.values()) / len(f1)}


def script_share(text: str, name: str) -> float:
    letters = [c for c in text if c.isalpha()]
    return sum(name in unicodedata.name(c, "") for c in letters) / max(len(letters), 1)


def main() -> None:
    from transformers import AutoModelForCausalLM, AutoTokenizer, get_cosine_schedule_with_warmup

    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--model", default="Qwen/Qwen3-0.6B")
    ap.add_argument("--out", required=True)
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--lr", type=float, default=2e-5)
    ap.add_argument("--bs", type=int, default=8)
    ap.add_argument("--accum", type=int, default=2)
    ap.add_argument("--max-len", type=int, default=1024)
    ap.add_argument("--max-new", type=int, default=200)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()

    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    random.seed(a.seed)
    torch.manual_seed(a.seed)
    rng = random.Random(a.seed)
    d = Path(a.data)
    raw = {s: read(d / f"{s}.jsonl") for s in ("train", "val", "test")}

    tok = AutoTokenizer.from_pretrained(a.model)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    enc, info = {}, {}
    for s, rows in raw.items():
        enc[s], dropped = encode(tok, rows, a.max_len)
        info[s] = {"examples": len(rows), "dropped_too_long": dropped}
    lens = sorted(len(x["ids"]) for x in enc["train"])
    info["train_tokens_median_p95_max"] = [lens[len(lens) // 2], lens[int(len(lens) * 0.95)], lens[-1]]
    print("data:", info, flush=True)

    model = AutoModelForCausalLM.from_pretrained(a.model, torch_dtype=torch.float32).cuda()
    n_params = sum(p.numel() for p in model.parameters())
    opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=0.01, betas=(0.9, 0.95))
    steps = a.epochs * math.ceil(len(enc["train"]) / (a.bs * a.accum))
    sched = get_cosine_schedule_with_warmup(opt, max(1, int(0.05 * steps)), steps)

    log, t0, step = [], time.time(), 0
    for ep in range(a.epochs):
        model.train()
        run, k = 0.0, 0
        for j, (ids, att, lab) in enumerate(batches(enc["train"], a.bs, tok.pad_token_id, True, rng)):
            with torch.autocast("cuda", dtype=torch.bfloat16):
                loss = model(input_ids=ids, attention_mask=att, labels=lab).loss
            (loss / a.accum).backward()
            run, k = run + loss.item(), k + 1
            if (j + 1) % a.accum == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                opt.step(), sched.step(), opt.zero_grad(set_to_none=True)
                step += 1
        vl = eval_loss(model, enc["val"], a.bs, tok.pad_token_id)
        log.append({"epoch": ep + 1, "train_loss": run / k, "val_loss": vl, "minutes": (time.time() - t0) / 60})
        print(log[-1], flush=True)

    model.save_pretrained(out / "model")
    tok.save_pretrained(out / "model")

    # Evaluate on the test split, restricted to examples that fit the token budget.
    keep =[r for r in raw["test"] if len(tok(r["input"] + SEP + r["output"], add_special_tokens=False)["input_ids"]) + 1 <= a.max_len]
    gens = generate(model, tok, keep, 16, a.max_new)
    pred = [parse(g) for g in gens]
    gold = [r["verdict"] for r in keep]
    pv = [p[0] for p in pred]
    acc = sum(g == p for g, p in zip(gold, pv)) / len(gold)
    majority = Counter(gold).most_common(1)[0]
    by = {}
    for key in ("source", "language"):
        for val in sorted({r[key] for r in keep}):
            idx = [i for i, r in enumerate(keep) if r[key] == val]
            by[f"{key}={val}"] = {"n": len(idx), "acc": sum(gold[i] == pv[i] for i in idx) / len(idx)}
    script = {}
    for lang, name in SCRIPTS.items():
        idx = [i for i, r in enumerate(keep) if r["language"] == lang and pred[i][1]]
        if idx:
            script[lang] = {"n": len(idx), "script_share": sum(script_share(pred[i][1], name) for i in idx) / len(idx)}
    metrics = {
        "model": a.model, "params_million": round(n_params / 1e6), "data": info, "train_log": log,
        "test_n": len(keep), "verdict_accuracy": acc, "majority_baseline": {"class": majority[0], "accuracy": majority[1] / len(gold)},
        "f1": macro_f1(gold, pv), "invalid_verdict_rate": pv.count("invalid") / len(pv),
        "confusion": {f"{g}->{p}": c for (g, p), c in sorted(Counter(zip(gold, pv)).items())},
        "by_group": by, "script_match": script,
        "narrative_words_median": sorted(len(p[1].split()) for p in pred)[len(pred) // 2],
    }
    (out / "metrics.json").write_text(json.dumps(metrics, indent=2, ensure_ascii=False))
    with (out / "test_generations.jsonl").open("w") as f:
        for r, g in zip(keep, gens):
            f.write(json.dumps({"input": r["input"], "reference": r["output"], "generated": g, "language": r["language"],
                                "source": r["source"]}, ensure_ascii=False) + "\n")
    print(json.dumps({k: metrics[k] for k in ("test_n", "verdict_accuracy", "majority_baseline", "f1", "invalid_verdict_rate",
                                                "confusion", "script_match", "narrative_words_median")}, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
