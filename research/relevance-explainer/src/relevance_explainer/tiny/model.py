"""A small decoder-only transformer sized for roughly 20 MB at INT8.

RMSNorm, rotary position embeddings, SwiGLU feed-forward, tied input and output embeddings.
Default configuration: vocab 8000, d_model 384, 10 layers, 6 heads, FFN 1024 -> about 20.8M parameters.
"""
from __future__ import annotations

import math
from dataclasses import asdict, dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class Config:
    vocab: int = 8000
    d_model: int = 384
    n_layer: int = 10
    n_head: int = 6
    d_ff: int = 1024
    max_len: int = 1280
    dropout: float = 0.1

    def to_dict(self) -> dict:
        return asdict(self)


class RMSNorm(nn.Module):
    def __init__(self, d: int, eps: float = 1e-6):
        super().__init__()
        self.w = nn.Parameter(torch.ones(d))
        self.eps = eps

    def forward(self, x):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps) * self.w


def rope_cache(head_dim: int, max_len: int, base: float = 10000.0):
    inv = 1.0 / (base ** (torch.arange(0, head_dim, 2).float() / head_dim))
    ang = torch.outer(torch.arange(max_len).float(), inv)
    return ang.cos(), ang.sin()


def apply_rope(x, cos, sin):
    # x: (B, H, T, D); cos/sin: (T, D/2) shared positions, or (B, T, D/2) per-sequence positions
    x1, x2 = x[..., 0::2], x[..., 1::2]
    if cos.dim() == 2:
        cos, sin = cos[None, None, : x.size(2)], sin[None, None, : x.size(2)]
    else:
        cos, sin = cos[:, None], sin[:, None]
    return torch.stack((x1 * cos - x2 * sin, x1 * sin + x2 * cos), dim=-1).flatten(-2)


class Block(nn.Module):
    def __init__(self, c: Config):
        super().__init__()
        self.n_head, self.dropout = c.n_head, c.dropout
        self.norm1, self.norm2 = RMSNorm(c.d_model), RMSNorm(c.d_model)
        self.qkv = nn.Linear(c.d_model, 3 * c.d_model, bias=False)
        self.proj = nn.Linear(c.d_model, c.d_model, bias=False)
        self.gate = nn.Linear(c.d_model, c.d_ff, bias=False)
        self.up = nn.Linear(c.d_model, c.d_ff, bias=False)
        self.down = nn.Linear(c.d_ff, c.d_model, bias=False)

    def forward(self, x, cos, sin, mask=None, cache=None):
        B, T, C = x.shape
        q, k, v = self.qkv(self.norm1(x)).view(B, T, 3, self.n_head, C // self.n_head).permute(2, 0, 3, 1, 4)
        q, k = apply_rope(q, cos, sin), apply_rope(k, cos, sin)
        if cache is not None:  # incremental decoding: keys and values of earlier tokens are kept
            if "k" in cache:
                k, v = torch.cat([cache["k"], k], dim=2), torch.cat([cache["v"], v], dim=2)
            cache["k"], cache["v"] = k, v
        if mask is None:
            a = F.scaled_dot_product_attention(q, k, v, is_causal=True, dropout_p=self.dropout if self.training else 0.0)
        else:
            a = F.scaled_dot_product_attention(q, k, v, attn_mask=mask)
        x = x + F.dropout(self.proj(a.transpose(1, 2).reshape(B, T, C)), self.dropout, self.training)
        h = self.norm2(x)
        return x + F.dropout(self.down(F.silu(self.gate(h)) * self.up(h)), self.dropout, self.training)


class TinyLM(nn.Module):
    def __init__(self, c: Config):
        super().__init__()
        self.c = c
        self.emb = nn.Embedding(c.vocab, c.d_model)
        self.blocks = nn.ModuleList(Block(c) for _ in range(c.n_layer))
        self.norm = RMSNorm(c.d_model)
        cos, sin = rope_cache(c.d_model // c.n_head, c.max_len)
        self.register_buffer("cos", cos, persistent=False)
        self.register_buffer("sin", sin, persistent=False)
        self.apply(self._init)
        for b in self.blocks:  # scale residual projections by depth
            for lin in (b.proj, b.down):
                nn.init.normal_(lin.weight, std=0.02 / math.sqrt(2 * c.n_layer))

    @staticmethod
    def _init(m):
        if isinstance(m, (nn.Linear, nn.Embedding)):
            nn.init.normal_(m.weight, std=0.02)

    def forward(self, ids, labels=None, weights=None):
        x = F.dropout(self.emb(ids), self.c.dropout, self.training)
        for b in self.blocks:
            x = b(x, self.cos, self.sin)
        logits = F.linear(self.norm(x), self.emb.weight)  # tied output projection
        if labels is None:
            return logits
        flat = logits[:, :-1].reshape(-1, logits.size(-1))
        target = labels[:, 1:].reshape(-1)
        if weights is None:
            return logits, F.cross_entropy(flat, target, ignore_index=-100)
        ce = F.cross_entropy(flat, target, ignore_index=-100, reduction="none")
        w = weights[:, 1:].reshape(-1) * (target != -100)
        return logits, (ce * w).sum() / w.sum().clamp_min(1e-8)

    @torch.no_grad()
    def generate(self, ids: torch.Tensor, eos: int, max_new: int, no_repeat: int = 4) -> list[int]:
        """Greedy decoding for one sequence (1, T); recomputes the full sequence each step.

        `no_repeat` > 0 forbids any token that would complete an n-gram of that size already generated."""
        out: list[int] = []
        seen: dict[tuple[int, ...], set[int]] = {}
        for _ in range(max_new):
            logits = self(ids[:, -self.c.max_len :])[0, -1]
            if no_repeat and len(out) >= no_repeat - 1:
                banned = seen.get(tuple(out[len(out) - no_repeat + 1 :]), ())
                if banned:
                    logits[list(banned)] = float("-inf")
            nxt = logits.argmax().item()
            if no_repeat and len(out) >= no_repeat - 1:
                seen.setdefault(tuple(out[len(out) - no_repeat + 1 :]), set()).add(nxt)
            if nxt == eos:
                break
            out.append(nxt)
            ids = torch.cat([ids, torch.tensor([[nxt]], device=ids.device)], dim=1)
        return out

    @torch.no_grad()
    def generate_batch(self, prompts: list[list[int]], eos: int, max_new: int, no_repeat: int = 4, pad: int = 0,
                       copy_bias: float = 0.0, copy_skip: int = 4) -> list[list[int]]:
        """Greedy decoding for several prompts at once with a KV cache.

        Prompts are left-padded; padding is masked out of attention and does not shift rotary positions,
        so each result equals single-sequence `generate` (see tests).

        `copy_bias` adds a bonus to the logit of every token that occurs in the prompt, from the `copy_skip`-th
        generated token on (the verdict line is left alone). It nudges the narrative towards terms of the input."""
        dev, B = self.emb.weight.device, len(prompts)
        L = max(len(p) for p in prompts)
        ids = torch.full((B, L), pad, dtype=torch.long, device=dev)
        valid = torch.zeros((B, L), dtype=torch.bool, device=dev)
        for i, p in enumerate(prompts):
            ids[i, L - len(p) :] = torch.tensor(p, device=dev)
            valid[i, L - len(p) :] = True
        pos = (valid.long().cumsum(1) - 1).clamp_min(0)
        # Causal and key-validity mask; the diagonal stays open so padded query rows never see an empty set.
        mask = torch.tril(torch.ones(L, L, dtype=torch.bool, device=dev))[None, None] & valid[:, None, None, :]
        mask = mask | torch.eye(L, dtype=torch.bool, device=dev)[None, None]
        caches: list[dict] = [{} for _ in self.blocks]
        x = self.emb(ids)
        for b, c in zip(self.blocks, caches):
            x = b(x, self.cos[pos], self.sin[pos], mask, c)
        logits = F.linear(self.norm(x[:, -1]), self.emb.weight).float()

        outs: list[list[int]] = [[] for _ in range(B)]
        seen: list[dict[tuple[int, ...], set[int]]] = [{} for _ in range(B)]
        done = [False] * B
        last = pos[:, -1]
        in_prompt = None
        if copy_bias:
            in_prompt = torch.zeros((B, self.c.vocab), dtype=torch.bool, device=dev)
            in_prompt.scatter_(1, ids, True)
            in_prompt[:, pad] = False
            in_prompt[:, eos] = False
        for step in range(max_new):
            if in_prompt is not None and step >= copy_skip:
                logits = logits + copy_bias * in_prompt
            if no_repeat:
                for i in range(B):
                    o = outs[i]
                    if not done[i] and len(o) >= no_repeat - 1:
                        banned = seen[i].get(tuple(o[len(o) - no_repeat + 1 :]), ())
                        if banned:
                            logits[i, list(banned)] = float("-inf")
            nxt = logits.argmax(-1).tolist()
            for i in range(B):
                if done[i]:
                    continue
                if nxt[i] == eos:
                    done[i] = True
                    continue
                o = outs[i]
                if no_repeat and len(o) >= no_repeat - 1:
                    seen[i].setdefault(tuple(o[len(o) - no_repeat + 1 :]), set()).add(nxt[i])
                o.append(nxt[i])
            if all(done):
                break
            last = last + 1
            valid = torch.cat([valid, torch.ones((B, 1), dtype=torch.bool, device=dev)], dim=1)
            step_mask = valid[:, None, None, :]
            x = self.emb(torch.tensor(nxt, device=dev)[:, None])
            for b, c in zip(self.blocks, caches):
                x = b(x, self.cos[last][:, None], self.sin[last][:, None], step_mask, c)
            logits = F.linear(self.norm(x[:, -1]), self.emb.weight).float()
        return outs

    def n_params(self) -> int:
        return sum(p.numel() for p in self.parameters())
