"""Group relative policy optimization (GRPO) of a trained task model's narratives, rewarded by the teacher judge.

Each step takes P training inputs, forces the teacher's verdict as a prefix (as in DPO), and samples G narratives per
input from the current model. The teacher judge scores every narrative (faithfulness, coherence, consistency with the
verdict, language); a small deterministic term penalizes numbers and names absent from the input, and truncated or
empty outputs get a fixed low reward without a judge call. Rewards are standardized within each group; groups whose
samples all score the same carry no signal and are skipped. The update is the clipped surrogate with a per-token KL
penalty to the starting model (k3 estimator), summed over tokens and divided by a constant (no length bias).

  TEACHER_URLS=http://localhost:8000/v1,http://localhost:8001/v1 CUDA_VISIBLE_DEVICES=1 \
  python -m relevance_explainer.tiny.grpo --run runs/f5-dpo --data data/f5 --out runs/f5-grpo --prompts 4000
"""
from __future__ import annotations

import argparse
import copy
import json
import random
import shutil
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import torch

from ..distill.onpolicy import score_one
from ..distill.teacher_checks import MODEL, post, urls
from ..synth.llm import parse_json
from .faithfulness import unsupported
from .judge import SYSTEM as JUDGE_SYSTEM
from .model import Config, TinyLM

SEP = "\n\n### Judgment\n"
EOS, PAD = 1, 0
FAIL = -2.0  # reward for an empty or truncated narrative

# The evaluation judge's rubric plus an informativeness score, so that saying less is not rewarded by itself.
INFORM_SYSTEM = JUDGE_SYSTEM.replace(
    "\n\nReturn only a JSON object:",
    "\n- informative (1-5): 5 = tells the searcher what the item contains and how it relates to their query and each "
    "filter; 3 = some of this, generic or vague in parts; 1 = says almost nothing specific about the item."
    "\n\nReturn only a JSON object:").replace('"language_ok": bool}', '"language_ok": bool, "informative": int}')


def score_informative(base: str, inp: str, verdict: str, narrative: str) -> dict | None:
    body = {"model": MODEL, "max_tokens": 100, "temperature": 0.0,
            "messages": [{"role": "system", "content": INFORM_SYSTEM},
                         {"role": "user", "content": f"INPUT:\n{inp}\n\nEXPLANATION:\n{verdict}\n{narrative}"}],
            "chat_template_kwargs": {"enable_thinking": False}}
    try:
        j = parse_json(post(base, body)["choices"][0]["message"]["content"])
        return {"faithful": int(j["faithful"]), "coherent": int(j["coherent"]), "consistent": bool(j["consistent"]),
                "language_ok": bool(j["language_ok"]), "informative": int(j["informative"])}
    except Exception:
        return None


def reward(score: dict | None, narrative: str, inp: str, truncated: bool, unsup_w: float, inf_w: float = 0.0) -> float | None:
    """Scalar reward for one narrative; None when the judge failed (the sample is then left out of its group)."""
    if truncated or not narrative.strip():
        return FAIL
    if score is None:
        return None
    r = 2.0 * (score["faithful"] - 1) / 4 + 0.5 * (score["coherent"] - 1) / 4
    r -= 0.0 if score["consistent"] else 1.0
    r -= 0.0 if score["language_ok"] else 1.0
    if inf_w and "informative" in score:
        r += inf_w * (score["informative"] - 1) / 4
    return r - unsup_w * min(len(unsupported(narrative, inp)), 4)


def advantages(groups: list[list[float | None]]) -> list[list[float | None]]:
    """Standardize rewards within each group. Groups with fewer than two scored samples or no spread give None."""
    out = []
    for g in groups:
        v = [x for x in g if x is not None]
        if len(v) < 2:
            out.append([None] * len(g))
            continue
        m = sum(v) / len(v)
        sd = (sum((x - m) ** 2 for x in v) / len(v)) ** 0.5
        out.append([None] * len(g) if sd < 1e-6 else [None if x is None else (x - m) / (sd + 1e-4) for x in g])
    return out


def token_logps(model: TinyLM, ids: torch.Tensor) -> torch.Tensor:
    """Log-probability of each next token, shape [B, T-1]."""
    with torch.autocast("cuda", dtype=torch.bfloat16):
        logits = model(ids)
    return torch.log_softmax(logits[:, :-1].float(), dim=-1).gather(-1, ids[:, 1:, None]).squeeze(-1)


def pack(seqs: list[tuple[list[int], list[int]]]) -> tuple[torch.Tensor, torch.Tensor]:
    """Right-padded ids and a mask over the completion targets, aligned with token_logps."""
    n = max(len(p) + len(c) for p, c in seqs)
    ids = torch.full((len(seqs), n), PAD)
    mask = torch.zeros((len(seqs), n - 1))
    for k, (p, c) in enumerate(seqs):
        ids[k, : len(p) + len(c)] = torch.tensor(p + c)
        mask[k, len(p) - 1 : len(p) + len(c) - 1] = 1.0
    return ids.cuda(), mask.cuda()


def sft_step(policy: TinyLM, batch: tuple[torch.Tensor, torch.Tensor], weight: float) -> float:
    """Backpropagate weight x mean NLL of the teacher's output tokens (verdict and narrative); returns the NLL."""
    ids, m = batch
    nll = -(token_logps(policy, ids) * m).sum() / m.sum()
    (weight * nll).backward()
    return nll.item()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True, help="starting model (also the KL reference)")
    ap.add_argument("--data", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--prompts", type=int, default=4000, help="training inputs used in total (one pass)")
    ap.add_argument("--per-step", type=int, default=16, help="inputs per step")
    ap.add_argument("--group", type=int, default=8, help="samples per input")
    ap.add_argument("--temp", type=float, default=0.9)
    ap.add_argument("--top-k", type=int, default=40)
    ap.add_argument("--max-new", type=int, default=200)
    ap.add_argument("--max-len", type=int, default=1280)
    ap.add_argument("--lr", type=float, default=1e-5)
    ap.add_argument("--beta", type=float, default=0.04, help="KL weight")
    ap.add_argument("--clip", type=float, default=0.2)
    ap.add_argument("--inner", type=int, default=2, help="updates per batch of samples")
    ap.add_argument("--micro", type=int, default=32)
    ap.add_argument("--norm-tokens", type=float, default=80.0, help="constant divisor of the summed token loss")
    ap.add_argument("--unsup-weight", type=float, default=0.25)
    ap.add_argument("--inf-weight", type=float, default=0.0, help="weight of the judge's informativeness score (0: not asked)")
    ap.add_argument("--sft", type=float, default=0.0, help="weight of the NLL on the teacher's full output for the step's inputs")
    ap.add_argument("--workers", type=int, default=128)
    ap.add_argument("--save-every", type=int, default=50)
    ap.add_argument("--seed", type=int, default=2)
    a = ap.parse_args()

    torch.manual_seed(a.seed)
    run, data, out = Path(a.run), Path(a.data), Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    from tokenizers import Tokenizer
    tk = Tokenizer.from_file(str(run / "tokenizer.json"))
    rows = [json.loads(l) for l in (data / "train.jsonl").read_text().splitlines()]
    order = random.Random(a.seed).sample(range(len(rows)), len(rows))
    prompts: list[tuple[int, list[int]]] = []
    for i in order:
        p = tk.encode(rows[i]["input"] + SEP + rows[i]["verdict"] + "\n").ids
        if len(p) + a.max_new <= a.max_len:
            prompts.append((i, p))
        if len(prompts) == a.prompts:
            break

    ck = torch.load(run / "model_fp32.pt", map_location="cpu")
    policy = TinyLM(Config(**ck["config"]))
    policy.load_state_dict(ck["state"])
    policy = policy.cuda().eval()  # dropout off throughout so old, new and reference log-probs are comparable
    ref = copy.deepcopy(policy).eval()
    for p in ref.parameters():
        p.requires_grad_(False)
    opt = torch.optim.AdamW(policy.parameters(), lr=a.lr, weight_decay=0.0, betas=(0.9, 0.95))
    steps = len(prompts) // a.per_step
    warm = max(1, steps // 20)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: min(1.0, (s + 1) / warm))
    servers = urls()
    pool = ThreadPoolExecutor(a.workers)
    log, t0 = [], time.time()
    sample_log = (out / "samples.jsonl").open("w")

    for step in range(steps):
        batch = prompts[step * a.per_step : (step + 1) * a.per_step]
        jobs = [(i, p) for i, p in batch for _ in range(a.group)]
        t_gen = time.time()
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
            gens = policy.generate_batch([p for _, p in jobs], EOS, a.max_new, temperature=a.temp, top_k=a.top_k)
        trunc = [len(g) >= a.max_new for g in gens]
        texts = [tk.decode(g).strip() for g in gens]
        t_judge = time.time()

        def judge(k: int) -> dict | None:
            if trunc[k] or not texts[k]:
                return None
            r = rows[jobs[k][0]]
            fn = score_informative if a.inf_weight else score_one
            return fn(servers[k % len(servers)], r["input"], r["verdict"], texts[k])

        scores = list(pool.map(judge, range(len(jobs))))
        rew = [reward(scores[k], texts[k], rows[jobs[k][0]]["input"], trunc[k], a.unsup_weight, a.inf_weight) for k in range(len(jobs))]
        groups = [rew[g * a.group : (g + 1) * a.group] for g in range(len(batch))]
        adv = [x for g in advantages(groups) for x in g]
        for k in range(len(jobs)):
            sample_log.write(json.dumps({"step": step, "i": jobs[k][0], "text": texts[k], "truncated": trunc[k],
                                         "score": scores[k], "reward": rew[k], "adv": adv[k]}, ensure_ascii=False) + "\n")
        sample_log.flush()

        use = [k for k in range(len(jobs)) if adv[k] is not None]
        t_upd = time.time()
        stats: dict[str, float] = {"pg": 0.0, "kl": 0.0, "clipped": 0.0}
        sft_batch = pack([(tk.encode(rows[i]["input"] + SEP).ids, tk.encode(rows[i]["output"]).ids + [EOS]) for i, _ in batch]) if a.sft else None
        if use:
            seqs = [(jobs[k][1], gens[k] + ([] if trunc[k] else [EOS])) for k in use]
            A = torch.tensor([adv[k] for k in use], device="cuda")
            chunks = [list(range(s, min(s + a.micro, len(use)))) for s in range(0, len(use), a.micro)]
            packed = [pack([seqs[j] for j in ch]) for ch in chunks]
            with torch.no_grad():
                old = [token_logps(policy, ids) for ids, _ in packed]
                refl = [token_logps(ref, ids) for ids, _ in packed]
            for _ in range(a.inner):
                for ch, (ids, m), o, r_ in zip(chunks, packed, old, refl):
                    lp = token_logps(policy, ids)
                    ratio = torch.exp(lp - o)
                    adv_t = A[ch][:, None]
                    pg = -torch.minimum(ratio * adv_t, ratio.clamp(1 - a.clip, 1 + a.clip) * adv_t)
                    kl = torch.exp(r_ - lp) - (r_ - lp) - 1
                    loss = ((pg + a.beta * kl) * m).sum() / (len(use) * a.norm_tokens)
                    loss.backward()
                    n_tok = m.sum().item()
                    stats["pg"] += (pg * m).sum().item() / n_tok / len(chunks) / a.inner
                    stats["kl"] += (kl * m).sum().item() / n_tok / len(chunks) / a.inner
                    stats["clipped"] += (((ratio - 1).abs() > a.clip).float() * m).sum().item() / n_tok / len(chunks) / a.inner
                if a.sft:
                    stats["sft_nll"] = stats.get("sft_nll", 0.0) + sft_step(policy, sft_batch, a.sft) / a.inner
                torch.nn.utils.clip_grad_norm_(policy.parameters(), 1.0)
                opt.step(), opt.zero_grad(set_to_none=True)
        sched.step()

        judged = [s for s in scores if s]
        if judged and "informative" in judged[0]:
            stats["informative"] = sum(s["informative"] for s in judged) / len(judged)
        log.append({"step": step + 1, "reward": sum(x for x in rew if x is not None) / max(1, sum(x is not None for x in rew)),
                    "faithful": sum(s["faithful"] for s in judged) / max(1, len(judged)),
                    "coherent": sum(s["coherent"] for s in judged) / max(1, len(judged)),
                    "truncated": sum(trunc) / len(trunc), "judge_fail": sum(1 for k in range(len(jobs)) if not trunc[k] and texts[k] and scores[k] is None) / len(jobs),
                    "groups_used": sum(1 for g in range(len(batch)) if adv[g * a.group] is not None or any(adv[g * a.group : (g + 1) * a.group])) / len(batch),
                    "samples_used": len(use) / len(jobs), **{k: round(v, 5) for k, v in stats.items()},
                    "sec_gen": round(t_judge - t_gen, 1), "sec_judge": round(t_upd - t_judge, 1), "sec_update": round(time.time() - t_upd, 1),
                    "minutes": round((time.time() - t0) / 60, 1)})
        print(json.dumps(log[-1]), flush=True)
        if (step + 1) % a.save_every == 0 or step + 1 == steps:
            torch.save({"config": ck["config"], "state": policy.state_dict()}, out / "model_fp32.pt")
            torch.save({"config": ck["config"], "state": policy.state_dict()}, out / f"model_step{step + 1}.pt")
            (out / "grpo_log.json").write_text(json.dumps({"args": vars(a), "steps": steps, "log": log}, indent=1))
    shutil.copy(run / "tokenizer.json", out / "tokenizer.json")
    (out / "grpo_log.json").write_text(json.dumps({"args": vars(a), "steps": steps, "log": log}, indent=1))
    print("GRPO-DONE", flush=True)


if __name__ == "__main__":
    main()
