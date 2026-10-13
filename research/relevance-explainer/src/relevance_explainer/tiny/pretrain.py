"""Pretrain the tiny LM on the tokenized multilingual corpus.

  CUDA_VISIBLE_DEVICES=1 python -m relevance_explainer.tiny.pretrain --root data/corpus --out runs/pre-a --tokens 3e8
Optional: --init-emb Qwen/Qwen3-0.6B initializes embeddings from Qwen (PCA to d_model, matched by token string).
Domain mixing: --domain-root data/corpus_domain --domain-share 0.3 draws that share of sequences from a second corpus
with the same layout (synth/domain_corpus.py), with its own held-out tail (DOMAIN_HOLDOUT tokens per language).

Learning-rate schedules: cosine (default) or warmup-stable-decay (--schedule wsd; Hu et al. 2024, Hagele et al. 2024):
constant after warmup, then linear decay to 0.1 x lr over the last --decay-frac of the run. With --save-every, the
stable phase writes resumable states (model, optimizer, step, data order); a decay branch from one of them is
  --resume-state runs/x/state-020000.pt --tokens <state tokens + decay tokens> --decay-start 20000 --out runs/x-d20k

Several GPUs: python -m torch.distributed.run --nproc-per-node 2 -m relevance_explainer.tiny.pretrain ... (data parallel; each GPU draws
--bs / world sequences per step from its own random stream, so --bs stays the global batch).
"""
from __future__ import annotations

import argparse
import datetime
import json
import math
import os
import time
from contextlib import nullcontext
from pathlib import Path

import numpy as np
import torch
from tokenizers import Tokenizer

from .corpus import LANGS
from .model import Config, TinyLM

HOLDOUT = 1_000_000  # tokens per language kept out of training
DOMAIN_HOLDOUT = 200_000  # the domain corpus is small: a shorter held-out tail per language


def load_shards(root: Path) -> dict[str, np.ndarray]:
    """Token arrays per language. Extra shards (tok/<lang>_*.bin) go in front of the Wikipedia shard, so the
    held-out slice (the last HOLDOUT tokens) is the same Wikipedia tail whether or not extra shards exist."""
    out = {}
    for l in LANGS:
        extra = [np.memmap(p, dtype=np.uint16, mode="r") for p in sorted((root / "tok").glob(f"{l}_*.bin"))]
        wiki = np.memmap(root / "tok" / f"{l}.bin", dtype=np.uint16, mode="r")
        out[l] = np.concatenate([*extra, wiki]) if extra else wiki
    return out


def load_all_shards(root: Path) -> dict[str, np.memmap]:
    """Every tok/<stem>.bin as its own language (the multilingual corpus of tiny/multicorpus.py)."""
    return {p.stem: np.memmap(p, dtype=np.uint16, mode="r") for p in sorted((root / "tok").glob("*.bin"))}


def sampler(shards: dict[str, np.memmap], alpha: float, seq: int, bs: int, rng: np.random.Generator, holdout: int = HOLDOUT,
            weights: dict[str, float] | None = None):
    """Batches of random windows; a language is drawn with probability proportional to its size**alpha, or to
    `weights` when given (the multilingual plan's budgets, which are already flattened and count repetition)."""
    langs = [l for l in shards if len(shards[l]) > holdout + seq + 1]
    if weights is not None:
        p = np.array([weights.get(l, 0.0) for l in langs], dtype=float)
    else:
        sizes = np.array([max(len(shards[l]) - holdout, 1) for l in langs], dtype=float)
        p = sizes**alpha
    p /= p.sum()

    def batch() -> torch.Tensor:
        out = np.empty((bs, seq), dtype=np.int64)
        for i, k in enumerate(rng.choice(len(langs), size=bs, p=p)):
            s = shards[langs[k]]
            o = int(rng.integers(0, len(s) - holdout - seq))
            out[i] = s[o : o + seq]
        return torch.from_numpy(out)

    return batch, dict(zip(langs, p.round(3)))


@torch.no_grad()
def heldout_ppl(model: TinyLM, shards, seq: int, n_seq: int = 64, tk=None, holdout: int = HOLDOUT, prefix: str = "") -> dict[str, float]:
    """Held-out perplexity per token for each language. With a tokenizer, also bits per UTF-8 byte of the same
    text (key '<lang>_bpb'), which is comparable across vocabularies. The first token of each sequence is not
    predicted, so bits per byte is slightly optimistic by the same small factor for every vocabulary."""
    model.eval()
    res = {}
    for l, s in shards.items():
        tail = np.asarray(s[-holdout:], dtype=np.int64)
        n_seq = min(n_seq, len(tail) // seq)
        x = torch.from_numpy(tail[: n_seq * seq].reshape(n_seq, seq)).cuda()
        tot = 0.0
        for i in range(0, n_seq, 16):
            with torch.autocast("cuda", dtype=torch.bfloat16):
                _, loss = model(x[i : i + 16], x[i : i + 16])
            tot += loss.item() * len(x[i : i + 16])
        res[prefix + l] = round(math.exp(tot / n_seq), 2)
        if tk is not None:
            nbytes = len(tk.decode(tail[: n_seq * seq].tolist()).encode("utf-8"))
            # tot / n_seq is the mean loss per predicted token; n_seq * (seq - 1) tokens are predicted in total
            res[f"{prefix}{l}_bpb"] = round(tot / n_seq * n_seq * (seq - 1) / math.log(2) / nbytes, 4)
    model.train()
    return res


def qwen_embeddings(tk: Tokenizer, qwen: str, d_model: int) -> torch.Tensor:
    """Embedding matrix for our vocabulary: average the Qwen embeddings of each token string, then PCA to d_model."""
    from safetensors import safe_open
    from huggingface_hub import hf_hub_download
    from transformers import AutoTokenizer

    qtk = AutoTokenizer.from_pretrained(qwen)
    with safe_open(hf_hub_download(qwen, "model.safetensors"), "pt") as f:
        E = f.get_tensor("model.embed_tokens.weight").float()
    bpe = qtk.backend_tokenizer.model
    out = torch.zeros(tk.get_vocab_size(), E.shape[1])
    missing = 0
    for tok, i in tk.get_vocab().items():
        if tok in ("<pad>", "<eos>"):
            continue
        ids = [t.id for t in bpe.tokenize(tok)]  # our tokens are already in the byte-level alphabet Qwen uses
        if ids:
            out[i] = E[ids].mean(0)
        else:
            missing += 1
    print(f"embedding init: {missing} tokens without a Qwen match", flush=True)
    out -= out.mean(0, keepdim=True)
    _, _, Vt = torch.linalg.svd(out, full_matrices=False)
    z = out @ Vt[:d_model].T
    return z / z.std() * 0.04


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--tokens", type=float, default=3e8)
    ap.add_argument("--seq", type=int, default=1024)
    ap.add_argument("--max-len", type=int, default=1536, help="rotary position table size; at least --seq (positions past --seq are not trained)")
    ap.add_argument("--bs", type=int, default=128)
    ap.add_argument("--micro", type=int, default=32, help="sequences per forward pass")
    ap.add_argument("--lr", type=float, default=2e-3)
    ap.add_argument("--alpha", type=float, default=0.5, help="language sampling exponent over corpus size")
    ap.add_argument("--d-model", type=int, default=384)
    ap.add_argument("--layers", type=int, default=10, help="layer applications per token")
    ap.add_argument("--unique", type=int, default=0, help="distinct blocks stored (0: no sharing)")
    ap.add_argument("--share", default="repeat", choices=["repeat", "cycle"])
    ap.add_argument("--heads", type=int, default=6)
    ap.add_argument("--d-ff", type=int, default=1024)
    ap.add_argument("--kv-heads", type=int, default=0, help="key/value heads (grouped-query attention); 0: one per head")
    ap.add_argument("--attn-gate", default="", choices=["", "head", "elem"], help="sigmoid gate on the attention output")
    ap.add_argument("--dropout", type=float, default=0.0)
    ap.add_argument("--init-emb", default=None)
    ap.add_argument("--resume", default=None, help="model.pt to continue pretraining from")
    ap.add_argument("--eval-every", type=int, default=500)
    ap.add_argument("--domain-root", default=None, help="second corpus (same layout) mixed in at --domain-share")
    ap.add_argument("--domain-share", type=float, default=0.0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--all-shards", action="store_true", help="every tok/*.bin is a language, weighted by plan.json")
    ap.add_argument("--holdout", type=int, default=HOLDOUT, help="tokens per language kept out of training")
    ap.add_argument("--eval-seqs", type=int, default=64, help="held-out sequences per language at each evaluation")
    ap.add_argument("--schedule", default="cosine", choices=["cosine", "wsd"])
    ap.add_argument("--decay-frac", type=float, default=0.2, help="wsd: share of the steps spent decaying")
    ap.add_argument("--decay-start", type=int, default=0, help="wsd: step where the decay starts (overrides --decay-frac)")
    ap.add_argument("--save-every", type=int, default=0, help="steps between resumable states (state-<step>.pt)")
    ap.add_argument("--resume-state", default=None, help="state-<step>.pt to continue the same run (or branch a decay)")
    a = ap.parse_args()

    world, rank = int(os.environ.get("WORLD_SIZE", 1)), int(os.environ.get("RANK", 0))
    if world > 1:  # started by torchrun: one process per GPU
        import torch.distributed as dist
        dist.init_process_group("nccl", timeout=datetime.timedelta(hours=1))  # rank 0 evaluates while the others wait
        torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))
    main0 = rank == 0
    if not main0:  # only the first process prints
        import builtins
        builtins.print = lambda *x, **k: None
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(a.seed)
    root = Path(a.root)
    tk = Tokenizer.from_file(str(root / "tokenizer.json"))
    weights = None
    if a.all_shards:  # multilingual corpus: every shard is a language, sampled by the plan's budgets
        shards = load_all_shards(root)
        plan = json.loads((root / "plan.json").read_text())
        weights = {k: v["chars"] for k, v in plan.items()}
    else:
        shards = load_shards(root)
    holdout = a.holdout
    rng = np.random.default_rng(a.seed + 1000 * rank)  # kept so a resumable state can restore the data order
    assert a.bs % world == 0 and (world == 1 or not a.domain_root), "--bs must split evenly; no domain mixing with torchrun"
    batch, probs = sampler(shards, a.alpha, a.seq, a.bs // world, rng, holdout, weights)
    print(f"{len(probs)} languages; language sampling:", probs, flush=True)
    eval_shards = {l: shards[l] for l in probs}
    dom = None
    if a.domain_root and a.domain_share > 0:  # each batch takes round(share * bs) sequences from the domain corpus
        dom = load_shards(Path(a.domain_root))
        n_dom = round(a.domain_share * a.bs)
        dbatch, dprobs = sampler(dom, a.alpha, a.seq, n_dom, np.random.default_rng(a.seed + 1), DOMAIN_HOLDOUT)
        gbatch, _ = sampler(shards, a.alpha, a.seq, a.bs - n_dom, np.random.default_rng(a.seed))
        batch = lambda: torch.cat([gbatch(), dbatch()])[torch.randperm(a.bs)]
        print(f"domain corpus: {n_dom} of {a.bs} sequences per batch; language sampling {dprobs}", flush=True)

    if a.resume:  # continue from a saved checkpoint (a further pass over the corpus at a lower learning rate)
        ck = torch.load(a.resume, map_location="cpu")
        cfg = Config(**{**ck["config"], "dropout": a.dropout})
        model = TinyLM(cfg)
        model.load_state_dict(ck["state"])
        model = model.cuda()
    else:
        cfg = Config(vocab=tk.get_vocab_size(), d_model=a.d_model, n_layer=a.layers, n_unique=a.unique, share=a.share,
                     n_head=a.heads, n_kv_head=a.kv_heads, attn_gate=a.attn_gate, d_ff=a.d_ff, max_len=a.max_len,
                     dropout=a.dropout)
        model = TinyLM(cfg).cuda()
        if a.init_emb:
            model.emb.weight.data.copy_(qwen_embeddings(tk, a.init_emb, a.d_model))
    print(f"params {model.n_params() / 1e6:.2f}M", flush=True)

    decay = [p for p in model.parameters() if p.dim() >= 2]
    nodecay = [p for p in model.parameters() if p.dim() < 2]
    opt = torch.optim.AdamW([{"params": decay, "weight_decay": 0.1}, {"params": nodecay, "weight_decay": 0.0}],
                            lr=a.lr, betas=(0.9, 0.95), fused=True)
    steps = int(a.tokens / (a.bs * a.seq))
    warm = min(300, steps // 10)
    if a.schedule == "wsd":
        decay0 = a.decay_start or int(steps * (1 - a.decay_frac))
        lr_at = lambda s: (s + 1) / warm if s < warm else 1.0 if s < decay0 else 1.0 - 0.9 * (s - decay0) / max(steps - decay0, 1)
    else:
        lr_at = lambda s: (s + 1) / warm if s < warm else 0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * (s - warm) / max(steps - warm, 1)))
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lr_at)

    log, t0, start = [], time.time(), 1
    if a.resume_state:  # same model, optimizer moments, step and data order as when the state was written
        st = torch.load(a.resume_state, map_location="cpu", weights_only=False)
        model.load_state_dict(st["state"])
        opt.load_state_dict(st["opt"])
        rngs = st["rng"] if isinstance(st["rng"], list) else [st["rng"]]  # one data stream per GPU
        assert len(rngs) == world, f"state written by {len(rngs)} processes, resumed with {world}"
        rng.bit_generator.state = rngs[rank]
        log, start = st["log"], st["step"] + 1
        for g in opt.param_groups:  # the schedule of this run decides the learning rate from here on
            g["lr"] = a.lr * lr_at(st["step"])
        sched = torch.optim.lr_scheduler.LambdaLR(opt, lr_at, last_epoch=st["step"] - 1)
        print(f"resumed {a.resume_state} at step {st['step']}", flush=True)
    net = model
    if world > 1:  # gradients are averaged over the GPUs; the wrapper also copies rank 0's initial weights to all
        net = torch.nn.parallel.DistributedDataParallel(model, device_ids=[int(os.environ["LOCAL_RANK"])])
    model.train()
    for step in range(start, steps + 1):
        xb = batch().cuda(non_blocking=True)
        parts = xb.split(a.micro)
        for i, x in enumerate(parts):  # gradient accumulation keeps the effective batch at --bs sequences
            with net.no_sync() if world > 1 and i < len(parts) - 1 else nullcontext():  # sync on the last part
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    _, loss = net(x, x)
                (loss * len(x) / len(xb)).backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step(), sched.step(), opt.zero_grad(set_to_none=True)
        if step % 100 == 0:
            print(f"step {step}/{steps} loss {loss.item():.3f} lr {sched.get_last_lr()[0]:.2e} {(time.time() - t0) / 60:.1f} min "
                  f"{(step - start + 1) * a.bs * a.seq / (time.time() - t0) / 1e3:.0f}k tok/s", flush=True)
        if (step % a.eval_every == 0 or step == steps) and main0:
            ppl = heldout_ppl(model, eval_shards, a.seq, n_seq=a.eval_seqs, tk=tk, holdout=holdout)
            bpb = [v for k, v in ppl.items() if k.endswith("_bpb")]
            print(f"mean held-out bits per byte over {len(bpb)} languages: {sum(bpb) / len(bpb):.4f}", flush=True)
            if dom is not None:
                ppl.update(heldout_ppl(model, dom, a.seq, n_seq=32, tk=tk, holdout=DOMAIN_HOLDOUT, prefix="domain_"))
            log.append({"step": step, "tokens": step * a.bs * a.seq, "train_loss": loss.item(), "heldout_ppl": ppl})
            print("heldout ppl:", ppl, flush=True)
            torch.save({"config": cfg.to_dict(), "state": model.state_dict()}, out / "model.pt")
            (out / "log.json").write_text(json.dumps({"args": vars(a), "params_m": model.n_params() / 1e6, "log": log}, indent=1))
        if a.save_every and step % a.save_every == 0:
            rngs = [None] * world
            if world > 1:
                dist.all_gather_object(rngs, rng.bit_generator.state)
            else:
                rngs = [rng.bit_generator.state]
            if main0:
                torch.save({"config": cfg.to_dict(), "state": model.state_dict(), "opt": opt.state_dict(), "step": step,
                            "rng": rngs, "log": log}, out / f"state-{step:06d}.pt")
        if world > 1 and (step % a.eval_every == 0 or step == steps or (a.save_every and step % a.save_every == 0)):
            dist.barrier()
    if world > 1:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
