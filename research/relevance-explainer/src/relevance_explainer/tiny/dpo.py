"""Direct preference optimization of a trained task model on (chosen, rejected) narratives for the same input.

The verdict is part of the prompt, so only the narrative is optimized. A frozen copy of the starting model is the
reference. A small supervised term on the chosen narrative keeps the text fluent.

  CUDA_VISIBLE_DEVICES=1 python -m relevance_explainer.tiny.dpo --run runs/f1-pre2 --data data/f1 --out runs/f1-dpo
"""
from __future__ import annotations

import argparse
import copy
import json
import random
import shutil
import time
from pathlib import Path

import torch
import torch.nn.functional as F
from tokenizers import Tokenizer

from .model import Config, TinyLM

SEP = "\n\n### Judgment\n"
EOS, PAD = 1, 0


def build(tk: Tokenizer, rows: list[dict], pairs: list[dict], max_len: int) -> list[dict]:
    out = []
    for p in pairs:
        r = rows[p["i"]]
        prompt = tk.encode(r["input"] + SEP + p["verdict"] + "\n").ids
        c, j = tk.encode(p["chosen"]).ids + [EOS], tk.encode(p["rejected"]).ids + [EOS]
        if len(prompt) + max(len(c), len(j)) <= max_len and p["chosen"] != p["rejected"]:
            out.append({"prompt": prompt, "chosen": c, "rejected": j})
    return out


def collate(items: list[dict]):
    """Rows [chosen_0..chosen_n-1, rejected_0..rejected_n-1]; mask marks the narrative tokens."""
    seqs = [(x["prompt"], x["chosen"]) for x in items] + [(x["prompt"], x["rejected"]) for x in items]
    n = max(len(p) + len(c) for p, c in seqs)
    ids = torch.full((len(seqs), n), PAD)
    mask = torch.zeros((len(seqs), n), dtype=torch.bool)
    for k, (p, c) in enumerate(seqs):
        ids[k, : len(p) + len(c)] = torch.tensor(p + c)
        mask[k, len(p) : len(p) + len(c)] = True
    return ids.cuda(), mask.cuda()


def seq_logps(model: TinyLM, ids: torch.Tensor, mask: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Summed log-probability of the masked tokens, and their count."""
    with torch.autocast("cuda", dtype=torch.bfloat16):
        logits = model(ids)
    lp = torch.log_softmax(logits[:, :-1].float(), dim=-1).gather(-1, ids[:, 1:, None]).squeeze(-1)
    m = mask[:, 1:].float()
    return (lp * m).sum(-1), m.sum(-1)


def dpo_loss(pc, pr, rc, rr, beta: float) -> tuple[torch.Tensor, torch.Tensor]:
    margin = beta * ((pc - pr) - (rc - rr))
    return -F.logsigmoid(margin).mean(), (margin > 0).float().mean()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--data", required=True)
    ap.add_argument("--pairs", default="dpo_pairs.jsonl")
    ap.add_argument("--out", required=True)
    ap.add_argument("--beta", type=float, default=0.1)
    ap.add_argument("--lr", type=float, default=1e-5)
    ap.add_argument("--epochs", type=int, default=2)
    ap.add_argument("--bs", type=int, default=16)
    ap.add_argument("--sft", type=float, default=0.2, help="weight of the per-token NLL on the chosen narrative")
    ap.add_argument("--max-len", type=int, default=1280)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()

    torch.manual_seed(a.seed)
    rng = random.Random(a.seed)
    run, data, out = Path(a.run), Path(a.data), Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    tk = Tokenizer.from_file(str(run / "tokenizer.json"))
    rows = [json.loads(l) for l in (data / "train.jsonl").read_text().splitlines()]
    pairs = [json.loads(l) for l in (data / a.pairs).read_text().splitlines()]
    items = build(tk, rows, pairs, a.max_len)
    print(f"{len(items)} usable pairs of {len(pairs)}", flush=True)

    ck = torch.load(run / "model_fp32.pt", map_location="cpu")
    policy = TinyLM(Config(**ck["config"]))
    policy.load_state_dict(ck["state"])
    policy = policy.cuda().eval()  # dropout off for both models so the log-ratios are comparable
    ref = copy.deepcopy(policy).eval()
    for p in ref.parameters():
        p.requires_grad_(False)

    opt = torch.optim.AdamW(policy.parameters(), lr=a.lr, weight_decay=0.0, betas=(0.9, 0.95))
    steps = a.epochs * (len(items) // a.bs)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: min(1.0, (s + 1) / max(1, steps // 10)) * (1 - s / max(steps, 1)))
    log, t0, step = [], time.time(), 0
    for ep in range(a.epochs):
        rng.shuffle(items)
        run_loss = run_acc = run_margin = 0.0
        nb = 0
        for s in range(0, len(items) - a.bs + 1, a.bs):
            ids, mask = collate(items[s : s + a.bs])
            with torch.no_grad():
                r_lp, _ = seq_logps(ref, ids, mask)
            p_lp, cnt = seq_logps(policy, ids, mask)
            n = a.bs
            loss, acc = dpo_loss(p_lp[:n], p_lp[n:], r_lp[:n], r_lp[n:], a.beta)
            nll = -(p_lp[:n] / cnt[:n]).mean()
            (loss + a.sft * nll).backward()
            torch.nn.utils.clip_grad_norm_(policy.parameters(), 1.0)
            opt.step(), sched.step(), opt.zero_grad(set_to_none=True)
            step += 1
            run_loss, run_acc, nb = run_loss + loss.item(), run_acc + acc.item(), nb + 1
            run_margin += ((p_lp[:n] - r_lp[:n]) - (p_lp[n:] - r_lp[n:])).mean().item()
        log.append({"epoch": ep + 1, "dpo_loss": run_loss / nb, "pref_acc": run_acc / nb, "log_ratio_margin": run_margin / nb,
                    "minutes": (time.time() - t0) / 60})
        print(log[-1], flush=True)

    torch.save({"config": ck["config"], "state": policy.state_dict()}, out / "model_fp32.pt")
    shutil.copy(run / "tokenizer.json", out / "tokenizer.json")
    (out / "dpo_log.json").write_text(json.dumps({"args": vars(a), "pairs": len(items), "log": log}, indent=1))


if __name__ == "__main__":
    main()
