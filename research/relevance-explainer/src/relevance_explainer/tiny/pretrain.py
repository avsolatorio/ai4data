"""Pretrain the tiny LM on the tokenized multilingual corpus.

  CUDA_VISIBLE_DEVICES=1 python -m relevance_explainer.tiny.pretrain --root data/corpus --out runs/pre-a --tokens 3e8
Optional: --init-emb Qwen/Qwen3-0.6B initializes embeddings from Qwen (PCA to d_model, matched by token string).
Domain mixing: --domain-root data/corpus_domain --domain-share 0.3 draws that share of sequences from a second corpus
with the same layout (synth/domain_corpus.py), with its own held-out tail (DOMAIN_HOLDOUT tokens per language).
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


def sampler(shards: dict[str, np.memmap], alpha: float, seq: int, bs: int, rng: np.random.Generator, holdout: int = HOLDOUT):
    langs = list(shards)
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
    ap.add_argument("--dropout", type=float, default=0.0)
    ap.add_argument("--init-emb", default=None)
    ap.add_argument("--resume", default=None, help="model.pt to continue pretraining from")
    ap.add_argument("--eval-every", type=int, default=500)
    ap.add_argument("--domain-root", default=None, help="second corpus (same layout) mixed in at --domain-share")
    ap.add_argument("--domain-share", type=float, default=0.0)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()

    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(a.seed)
    root = Path(a.root)
    tk = Tokenizer.from_file(str(root / "tokenizer.json"))
    shards = load_shards(root)
    batch, probs = sampler(shards, a.alpha, a.seq, a.bs, np.random.default_rng(a.seed))
    print("language sampling:", probs, flush=True)
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
                     n_head=a.heads, d_ff=a.d_ff, max_len=1536, dropout=a.dropout)
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
    lr_at = lambda s: (s + 1) / warm if s < warm else 0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * (s - warm) / max(steps - warm, 1)))
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lr_at)

    log, t0 = [], time.time()
    model.train()
    for step in range(1, steps + 1):
        xb = batch().cuda(non_blocking=True)
        for x in xb.split(a.micro):  # gradient accumulation keeps the effective batch at --bs sequences
            with torch.autocast("cuda", dtype=torch.bfloat16):
                _, loss = model(x, x)
            (loss * len(x) / len(xb)).backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step(), sched.step(), opt.zero_grad(set_to_none=True)
        if step % 100 == 0:
            print(f"step {step}/{steps} loss {loss.item():.3f} {(time.time() - t0) / 60:.1f} min "
                  f"{step * a.bs * a.seq / (time.time() - t0) / 1e3:.0f}k tok/s", flush=True)
        if step % a.eval_every == 0 or step == steps:
            ppl = heldout_ppl(model, shards, a.seq, tk=tk)
            if dom is not None:
                ppl.update(heldout_ppl(model, dom, a.seq, n_seq=32, tk=tk, holdout=DOMAIN_HOLDOUT, prefix="domain_"))
            log.append({"step": step, "tokens": step * a.bs * a.seq, "train_loss": loss.item(), "heldout_ppl": ppl})
            print("heldout ppl:", ppl, flush=True)
            torch.save({"config": cfg.to_dict(), "state": model.state_dict()}, out / "model.pt")
            (out / "log.json").write_text(json.dumps({"args": vars(a), "params_m": model.n_params() / 1e6, "log": log}, indent=1))


if __name__ == "__main__":
    main()
