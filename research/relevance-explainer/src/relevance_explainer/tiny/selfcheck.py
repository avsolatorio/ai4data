"""Correctness head: a small classifier on the frozen explainer's own hidden states that predicts whether an output is
correct (verdict agrees with the teacher; explanation judged faithful, coherent, consistent and in the right language).

  sample    the deployed model writes its greedy output and K sampled explanations (same verdict) per input; the
            generated token ids are kept so features read exactly the states computed during generation
  score     the teacher judges every explanation (same judge as on-policy DPO)
  features  one teacher-forced pass of the frozen model over prompt + generated tokens: final hidden state at the last
            output token, mean hidden state over the output, hidden state at the verdict position, and scalar
            uncertainty features (verdict probability, token log-probabilities, entropy, invented numbers)
  train     fit heads on one set's features and report them on another, against the scalar baselines

The hidden states used here are the ones the model computes while generating (the last output token's state is the
one that predicts end of text), so in deployment the head adds one small matrix product per output.
"""
from __future__ import annotations

import argparse
import json
import math
import random
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from tokenizers import Tokenizer

from .model import Config, TinyLM
from .numguard import unsupported
from .scriptguard import banned_for, token_masks
from .train import quantize

SEP = "\n\n### Judgment\n"
EOS = 1
VERDICTS = ("relevant", "partly_relevant", "irrelevant")


def _read(p: Path) -> list[dict]:
    return [json.loads(l) for l in p.read_text().splitlines()]


def load_model(run: Path, device: str = "cuda") -> tuple[TinyLM, Tokenizer]:
    """The run's model quantized to INT8 as deployed (same path as tiny.evaluate)."""
    ck = torch.load(run / "model_fp32.pt", map_location="cpu")
    model = TinyLM(Config(**ck["config"]))
    model.load_state_dict(ck["state"])
    model, _ = quantize(model, 8)
    return model.to(device).eval(), Tokenizer.from_file(str(run / "tokenizer.json"))


# ------------------------------------------------------------------------------------------------------------ sample
def cmd_sample(a) -> None:
    out = Path(a.out)
    if out.exists():
        print(f"{out} exists")
        return
    model, tk = load_model(Path(a.run))
    rows = _read(Path(a.data) / f"{a.split}.jsonl")
    idx = sorted(random.Random(a.seed).sample(range(len(rows)), min(a.n, len(rows))))
    masks = token_masks(tk)
    prompts = {i: tk.encode(rows[i]["input"] + SEP).ids for i in idx}
    idx = [i for i in idx if len(prompts[i]) + a.max_new <= model.c.max_len]
    order = sorted(idx, key=lambda i: len(prompts[i]))
    greedy: dict[int, list[int]] = {}
    for s in range(0, len(order), a.batch):  # greedy output, as the app writes it before any number resampling
        ch = order[s : s + a.batch]
        ban = torch.stack([banned_for(masks, rows[i]["input"]) for i in ch])
        with torch.autocast("cuda", dtype=torch.bfloat16):
            outs = model.generate_batch([prompts[i] for i in ch], EOS, a.max_new, banned=ban)
        greedy.update(zip(ch, outs))
    # K sampled explanations after the greedy verdict line: the candidates a retry would choose from
    vline = {}
    for i in idx:
        first = tk.decode(greedy[i]).split("\n")[0].strip()
        vline[i] = tk.encode(rows[i]["input"] + SEP + first + "\n").ids
    torch.manual_seed(a.seed)
    samples: dict[int, list[list[int]]] = {i: [] for i in idx}
    jobs = [i for i in order for _ in range(a.k)]
    for s in range(0, len(jobs), a.batch):
        ch = jobs[s : s + a.batch]
        ban = torch.stack([banned_for(masks, rows[i]["input"]) for i in ch])
        with torch.autocast("cuda", dtype=torch.bfloat16):
            outs = model.generate_batch([vline[i] for i in ch], EOS, a.max_new, banned=ban, temperature=a.temp)
        for i, o in zip(ch, outs):
            samples[i].append(vline[i][len(prompts[i]) :] + o)  # verdict line tokens + sampled explanation
    with out.open("w") as f:
        for i in idx:
            r = rows[i]
            cands = [greedy[i]] + samples[i]
            f.write(json.dumps({"i": i, "label": r["verdict"], "unanimous": r.get("unanimous"), "language": r["language"],
                                "source": r.get("source"), "ids": cands, "texts": [tk.decode(c).strip() for c in cands]},
                               ensure_ascii=False) + "\n")
    print(f"sampled {len(idx)} inputs x {1 + a.k} outputs -> {out}")


# ------------------------------------------------------------------------------------------------------------- score
def cmd_score(a) -> None:
    from ..distill.onpolicy import score_one
    from ..distill.teacher_checks import urls
    out = Path(a.out)
    rows = _read(Path(a.data) / f"{a.split}.jsonl")
    samp = _read(Path(a.samples))
    done = {d["i"] for d in _read(out)} if out.exists() else set()
    todo = [d for d in samp if d["i"] not in done]
    servers = urls()

    def one(job):
        i, k, text = job
        verdict, _, narr = text.partition("\n")
        return score_one(servers[(i + k) % len(servers)], rows[i]["input"], verdict.strip(), narr.strip())

    for s in range(0, len(todo), 1000):  # in chunks, appended, so an interruption keeps finished work
        ch = todo[s : s + 1000]
        jobs = [(d["i"], k, t) for d in ch for k, t in enumerate(d["texts"])]
        with ThreadPoolExecutor(a.workers) as ex:
            res = list(ex.map(one, jobs))
        by: dict[int, list] = {d["i"]: [None] * len(d["texts"]) for d in ch}
        for (i, k, _), r in zip(jobs, res):
            by[i][k] = r
        with out.open("a") as f:
            for d in ch:
                f.write(json.dumps({"i": d["i"], "scores": by[d["i"]]}) + "\n")
        print(f"scored {s + len(ch)} / {len(todo)} inputs", flush=True)


# ---------------------------------------------------------------------------------------------------------- features
@torch.no_grad()
def output_features(model: TinyLM, prompt: list[int], out: list[int], vfirst: list[int]) -> dict:
    """Hidden states and uncertainty scalars of one generated output, from one pass over prompt + output.

    Position t's state predicts token t+1, so the last prompt position predicts the verdict and the last output position
    predicts end of text: both are states the model computed while generating."""
    dev = model.emb.weight.device
    ids = torch.tensor([prompt + out], device=dev)
    x = model.emb(ids)
    for j in model.order:
        x = model.blocks[j](x, model.cos, model.sin, None)
    h = model.norm(x)[0].float()
    logp = F.log_softmax(F.linear(h, model.emb.weight.float()), -1)
    P = len(prompt)
    pos = torch.arange(P - 1, P - 1 + len(out), device=dev)  # positions that predicted each output token
    tgt = ids[0, P:]
    tok_lp = logp[pos, tgt]
    ent = -(logp[pos].exp() * logp[pos]).sum(-1)
    nl = out.index(tk_newline) if tk_newline in out else 0  # explanation tokens start after the verdict line
    narr_lp, narr_ent = tok_lp[nl + 1 :], ent[nl + 1 :]
    if narr_lp.numel() == 0:
        narr_lp, narr_ent = tok_lp, ent
    v = logp[P - 1, vfirst].softmax(0)  # first tokens of the three verdicts are distinct
    last = P - 1 + len(out)  # the state after the last output token, which predicted end of text
    return {
        "h_last": h[min(last, h.size(0) - 1)].half().cpu(),
        "h_mean": h[P : P + len(out)].mean(0).half().cpu(),
        "h_verdict": h[P - 1].half().cpu(),
        "scalars": [v.max().item(), narr_lp.mean().item(), narr_lp.min().item(), narr_ent.mean().item(),
                    narr_ent.max().item(), math.log1p(len(out)), logp[min(last, h.size(0) - 1), EOS].exp().item()],
    }


SCALARS = ["verdict_prob", "mean_logp", "min_logp", "mean_entropy", "max_entropy", "log_length", "p_eos"]
tk_newline = -1


def cmd_features(a) -> None:
    global tk_newline
    out = Path(a.out)
    if out.exists():
        print(f"{out} exists")
        return
    model, tk = load_model(Path(a.run))
    tk_newline = tk.encode("\n").ids[-1]
    vfirst = [tk.encode(v).ids[0] for v in VERDICTS]
    assert len(set(vfirst)) == 3
    rows = _read(Path(a.data) / f"{a.split}.jsonl")
    samp = _read(Path(a.samples))
    scores = {d["i"]: d["scores"] for d in _read(Path(a.scored))} if a.scored else {}
    recs = {k: [] for k in ("h_last", "h_mean", "h_verdict", "scalars")}
    meta = []
    for d in samp:
        i, inp = d["i"], rows[d["i"]]["input"]
        prompt = tk.encode(inp + SEP).ids
        for k, (ids, text) in enumerate(zip(d["ids"], d["texts"])):
            with torch.autocast("cuda", dtype=torch.bfloat16):
                f = output_features(model, prompt, ids, vfirst)
            pred = text.split("\n")[0].strip()
            invented = bool(unsupported(text, inp))
            for key in recs:
                recs[key].append(torch.tensor(f[key]) if key == "scalars" else f[key])
            recs["scalars"][-1] = torch.cat([recs["scalars"][-1], torch.tensor([float(invented)])])
            s = (scores.get(i) or [None] * len(d["texts"]))[k]
            meta.append({"i": i, "k": k, "pred": pred, "label": d["label"], "unanimous": d.get("unanimous"),
                         "language": d["language"], "source": d.get("source"), "invented": invented, "judge": s})
    torch.save({"h_last": torch.stack(recs["h_last"]), "h_mean": torch.stack(recs["h_mean"]),
                "h_verdict": torch.stack(recs["h_verdict"]), "scalars": torch.stack(recs["scalars"]).float(),
                "scalar_names": SCALARS + ["invented_number"], "meta": meta}, out)
    print(f"features of {len(meta)} outputs -> {out}")


# ------------------------------------------------------------------------------------------------------------- train
def targets(meta: list[dict]) -> dict[str, torch.Tensor]:
    """verdict: the verdict matches the teacher's label. explanation: judged faithful >= 4, coherent >= 4, consistent
    with its verdict, in the requested language, and no number absent from the input. both: the two together."""
    v = torch.tensor([m["pred"] == m["label"] for m in meta], dtype=torch.float)
    e = torch.tensor([bool(m["judge"]) and m["judge"]["faithful"] >= 4 and m["judge"]["coherent"] >= 4
                      and m["judge"]["consistent"] and m["judge"]["language_ok"] and not m["invented"] for m in meta],
                     dtype=torch.float)
    return {"verdict": v, "explanation": e, "both": v * e}


def inputs_of(d: dict, kind: str, mu=None, sd=None):
    s = d["scalars"]
    if mu is None:
        mu, sd = s.mean(0), s.std(0).clamp_min(1e-6)
    s = (s - mu) / sd
    parts = {"scalars": [s], "hidden": [d["h_last"].float(), d["h_mean"].float(), d["h_verdict"].float()],
             "hidden+scalars": [d["h_last"].float(), d["h_mean"].float(), d["h_verdict"].float(), s]}[kind]
    return torch.cat(parts, 1), mu, sd


class Head(nn.Module):
    def __init__(self, d_in: int, hidden: int):
        super().__init__()
        self.net = nn.Linear(d_in, 1) if hidden == 0 else nn.Sequential(nn.Linear(d_in, hidden), nn.GELU(),
                                                                         nn.Dropout(0.1), nn.Linear(hidden, 1))

    def forward(self, x):
        return self.net(x).squeeze(-1)


def fit(x: torch.Tensor, y: torch.Tensor, groups: list[int], hidden: int, seed: int = 0) -> Head:
    """BCE with AdamW; 10% of inputs (all outputs of an input together) held out for early stopping."""
    torch.manual_seed(seed)
    ug = sorted(set(groups))
    dev_g = set(random.Random(seed).sample(ug, max(1, len(ug) // 10)))
    dev_m = torch.tensor([g in dev_g for g in groups])
    xt, yt, xd, yd = x[~dev_m], y[~dev_m], x[dev_m], y[dev_m]
    head = Head(x.size(1), hidden)
    opt = torch.optim.AdamW(head.parameters(), lr=1e-3, weight_decay=0.01)
    best, best_state, bad = float("inf"), None, 0
    for ep in range(200):
        head.train()
        perm = torch.randperm(len(xt))
        for s in range(0, len(xt), 256):
            b = perm[s : s + 256]
            loss = F.binary_cross_entropy_with_logits(head(xt[b]), yt[b])
            opt.zero_grad()
            loss.backward()
            opt.step()
        head.eval()
        with torch.no_grad():
            dl = F.binary_cross_entropy_with_logits(head(xd), yd).item()
        if dl < best - 1e-4:
            best, best_state, bad = dl, {k: v.clone() for k, v in head.state_dict().items()}, 0
        else:
            bad += 1
            if bad >= 10:
                break
    head.load_state_dict(best_state)
    return head.eval()


def auroc(p: torch.Tensor, y: torch.Tensor) -> float:
    pos, neg = p[y == 1], p[y == 0]
    if len(pos) == 0 or len(neg) == 0:
        return float("nan")
    r = torch.cat([pos, neg]).argsort().argsort().float() + 1  # ranks (ties broken arbitrarily)
    return ((r[: len(pos)].sum() - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg))).item()


def ece(p: torch.Tensor, y: torch.Tensor, bins: int = 10) -> float:
    e = 0.0
    for b in range(bins):
        m = (p >= b / bins) & (p < (b + 1) / bins if b < bins - 1 else p <= 1)
        if m.any():
            e += m.float().mean().item() * abs(p[m].mean().item() - y[m].mean().item())
    return e


def selective(p: torch.Tensor, y: torch.Tensor, cov: float) -> float:
    k = max(1, int(len(p) * cov))
    return y[p.argsort(descending=True)[:k]].mean().item()


def report(name: str, p: torch.Tensor, y: torch.Tensor) -> dict:
    r = {"auroc": auroc(p, y), "ece": ece(p, y), "brier": ((p - y) ** 2).mean().item(),
         "acc@100": y.mean().item(), "acc@80": selective(p, y, 0.8), "acc@50": selective(p, y, 0.5)}
    print(f"  {name:34s} AUROC {r['auroc']:.3f}  ECE {r['ece']:.3f}  Brier {r['brier']:.3f}  "
          f"correct: all {r['acc@100']:.3f}, top 80% {r['acc@80']:.3f}, top 50% {r['acc@50']:.3f}")
    return r


def cmd_train(a) -> None:
    tr, te = torch.load(a.train_features, weights_only=False), torch.load(a.test_features, weights_only=False)
    keep_tr = [j for j, m in enumerate(tr["meta"]) if m["judge"]]  # outputs the judge failed on are left out
    keep_te = [j for j, m in enumerate(te["meta"]) if m["judge"]]
    sub = lambda d, keep: {k: (v[keep] if torch.is_tensor(v) else [v[j] for j in keep]) for k, v in d.items()  # noqa: E731
                           if k != "scalar_names"}
    tr, te = sub(tr, keep_tr), sub(te, keep_te)
    ytr, yte = targets(tr["meta"]), targets(te["meta"])
    groups = [m["i"] for m in tr["meta"]]
    greedy = torch.tensor([m["k"] == 0 for m in te["meta"]])
    results, heads, preds = {}, {}, {}
    for target in ("verdict", "explanation", "both"):
        print(f"\n== target: {target} (train positive rate {ytr[target].mean():.3f}, n={len(groups)}; "
              f"test greedy outputs n={int(greedy.sum())})")
        res = {}
        vp = te["scalars"][:, 0]
        res["verdict probability"] = report("verdict probability (no training)", vp[greedy], yte[target][greedy])
        mlp = te["scalars"][:, 1]
        res["mean token log-prob"] = report("mean token log-prob (no training)", mlp[greedy].exp(), yte[target][greedy])
        for kind, hidden in (("scalars", 0), ("hidden", 0), ("hidden", 64), ("hidden+scalars", 64)):
            xtr, mu, sd = inputs_of(tr, kind)
            xte, _, _ = inputs_of(te, kind, mu, sd)
            head = fit(xtr, ytr[target], groups, hidden)
            with torch.no_grad():
                p = torch.sigmoid(head(xte))
            nm = f"{kind}, {'linear' if hidden == 0 else f'MLP {hidden}'} ({sum(q.numel() for q in head.parameters()):,} params)"
            res[nm] = report(nm, p[greedy], yte[target][greedy])
            heads[(target, kind, hidden)] = (head, mu, sd)
            preds[(target, kind, hidden)] = p
        results[target] = res
    # retry: among the greedy output and its K sampled explanations (same verdict), keep the one the head rates highest
    print("\n== retry by the explanation head (hidden+scalars, MLP 64): judged quality of the kept explanation")
    p = preds[("explanation", "hidden+scalars", 64)]
    by: dict[int, list[int]] = {}
    for j, m in enumerate(te["meta"]):
        by.setdefault(m["i"], []).append(j)
    rows = {"greedy": [], "head best of all": [], "greedy unless below threshold": [], "random sample": []}
    rng = random.Random(0)
    for i, js in by.items():
        g = next((j for j in js if te["meta"][j]["k"] == 0), None)
        if g is None or len(js) < 2:
            continue
        rows["greedy"].append(g)
        rows["head best of all"].append(max(js, key=lambda j: p[j].item()))
        rows["greedy unless below threshold"].append(g if p[g] >= a.threshold else max(js, key=lambda j: p[j].item()))
        rows["random sample"].append(rng.choice([j for j in js if j != g]))
    retry = {}
    for nm, js in rows.items():
        f = [te["meta"][j]["judge"]["faithful"] for j in js]
        ok = yte["explanation"][js].mean().item()
        retry[nm] = {"n": len(js), "faithful": sum(f) / len(f), "faithful_ge4": sum(x >= 4 for x in f) / len(f), "good": ok}
        print(f"  {nm:32s} n {len(js)}  faithful {retry[nm]['faithful']:.3f}  faithful>=4 {retry[nm]['faithful_ge4']:.3f}  "
              f"good {ok:.3f}")
    oracle = [max(js, key=lambda j: (yte['explanation'][j].item(), te['meta'][j]['judge']['faithful'])) for i, js in by.items()
              if len(js) > 1]
    print(f"  {'oracle (best judged of all)':32s} good {yte['explanation'][oracle].mean().item():.3f}")
    retry["retried_share"] = sum(1 for g in rows["greedy"] if p[g] < a.threshold) / max(1, len(rows["greedy"]))
    print(f"  share retried at threshold {a.threshold}: {retry['retried_share']:.3f}")
    if a.out:
        Path(a.out).write_text(json.dumps({"results": results, "retry": retry}, indent=1))
    if a.save_head:
        head, mu, sd = heads[("both", "hidden+scalars", 64)]
        torch.save({"state": head.state_dict(), "mu": mu, "sd": sd, "d_in": head.net[0].in_features, "hidden": 64,
                    "target": "both", "features": "h_last,h_mean,h_verdict,scalars", "scalar_names": SCALARS + ["invented_number"]},
                   a.save_head)
        print(f"saved head -> {a.save_head}")


def main() -> None:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("sample")
    p.add_argument("--run", required=True); p.add_argument("--data", required=True); p.add_argument("--split", default="val")
    p.add_argument("--n", type=int, default=6000); p.add_argument("--k", type=int, default=3)
    p.add_argument("--temp", type=float, default=0.7); p.add_argument("--max-new", type=int, default=260)
    p.add_argument("--batch", type=int, default=48); p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", required=True); p.set_defaults(fn=cmd_sample)
    p = sub.add_parser("score")
    p.add_argument("--data", required=True); p.add_argument("--split", default="val"); p.add_argument("--samples", required=True)
    p.add_argument("--out", required=True); p.add_argument("--workers", type=int, default=160); p.set_defaults(fn=cmd_score)
    p = sub.add_parser("features")
    p.add_argument("--run", required=True); p.add_argument("--data", required=True); p.add_argument("--split", default="val")
    p.add_argument("--samples", required=True); p.add_argument("--scored", default=None)
    p.add_argument("--out", required=True); p.set_defaults(fn=cmd_features)
    p = sub.add_parser("train")
    p.add_argument("--train-features", required=True); p.add_argument("--test-features", required=True)
    p.add_argument("--threshold", type=float, default=0.5); p.add_argument("--out", default=None)
    p.add_argument("--save-head", default=None); p.set_defaults(fn=cmd_train)
    a = ap.parse_args()
    a.fn(a)


if __name__ == "__main__":
    main()
