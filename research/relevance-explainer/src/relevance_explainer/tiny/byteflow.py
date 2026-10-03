"""ByteFlow Net (Deng et al., ICLR 2026, arXiv 2603.03583) at about 20M parameters.

Stages, as in the paper (section 3): byte embedding -> local encoder (pre-norm blocks with sliding-window attention
and Canon layers) -> coding-rate chunking (Top-K marginal coding rates, position 0 always kept) -> global transformer
over the K selected positions -> upsampler (bin-shared projections plus a residual on the local representation) ->
local decoder (same block type as the encoder) -> next-byte softmax over 258 symbols (256 bytes, BOS, EOS).

Two choices the paper leaves open or defines only for teacher forcing:
- The upsampler bins index the position within the current chunk (capped at `bins - 1`), where the paper bins the
  absolute position by sequence length. Chunk-relative bins have the same cost and are well defined while generating.
- The paper evaluates teacher-forced only. For generation, boundaries after the prompt are decided with a threshold:
  the smallest marginal coding rate among the prompt's own Top-K selection.
"""
from __future__ import annotations

import math
from dataclasses import asdict, dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

from .model import Block, Config as GlobalConfig, RMSNorm, apply_rope, rope_cache

BOS, EOS, VOCAB = 256, 257, 258


@dataclass
class ByteConfig:
    d_local: int = 192
    n_local: int = 2  # blocks in the encoder and, separately, in the decoder
    local_heads: int = 3
    local_ff: int = 512
    window: int = 128
    d_global: int = 448
    n_global: int = 7
    global_heads: int = 7
    global_ff: int = 1024
    ratio: float = 3.0  # bytes per global position: K = ceil(T / ratio)
    bins: int = 16
    rate: str = "logdet"  # "logdet" (exact marginal coding rate) or "l2" (the paper's norm approximation)
    max_bytes: int = 6144
    max_global: int = 2048
    dropout: float = 0.0

    def to_dict(self) -> dict:
        return asdict(self)


class Canon(nn.Module):
    """Causal depthwise convolution of kernel 4 with learned per-channel gates (Allen-Zhu, 2025); identity at init."""

    def __init__(self, d: int):
        super().__init__()
        self.w = nn.Parameter(torch.zeros(4, d))
        with torch.no_grad():
            self.w[0] = 1.0

    def forward(self, x: torch.Tensor, cache: torch.Tensor | None = None) -> tuple[torch.Tensor, torch.Tensor]:
        prev = cache if cache is not None else x.new_zeros(x.size(0), 3, x.size(2))
        xp = torch.cat([prev, x], dim=1)
        T = x.size(1)
        out = self.w[0] * xp[:, 3 : 3 + T] + self.w[1] * xp[:, 2 : 2 + T] + self.w[2] * xp[:, 1 : 1 + T] + self.w[3] * xp[:, 0:T]
        return out, xp[:, -3:]


def band_mask(T: int, window: int, device) -> torch.Tensor:
    """Causal mask where position i attends to j with i - window < j <= i."""
    i = torch.arange(T, device=device)[:, None]
    j = torch.arange(T, device=device)[None, :]
    return (j <= i) & (j > i - window)


class LocalBlock(nn.Module):
    """Equations 6 to 9: Canon(h + SWA(LN h)), then Canon(h + SwiGLU(LN h))."""

    def __init__(self, c: ByteConfig):
        super().__init__()
        d, self.n_head, self.window = c.d_local, c.local_heads, c.window
        self.norm1, self.norm2 = RMSNorm(d), RMSNorm(d)
        self.qkv = nn.Linear(d, 3 * d, bias=False)
        self.proj = nn.Linear(d, d, bias=False)
        self.gate = nn.Linear(d, c.local_ff, bias=False)
        self.up = nn.Linear(d, c.local_ff, bias=False)
        self.down = nn.Linear(c.local_ff, d, bias=False)
        self.canon1, self.canon2 = Canon(d), Canon(d)

    def forward(self, x, cos, sin, cache: dict | None = None):
        B, T, C = x.shape
        q, k, v = self.qkv(self.norm1(x)).view(B, T, 3, self.n_head, C // self.n_head).permute(2, 0, 3, 1, 4)
        q, k = apply_rope(q, cos, sin), apply_rope(k, cos, sin)
        if cache is None or "k" not in cache:
            mask = band_mask(T, self.window, x.device)[None, None] if T > 1 else None
        else:  # incremental step: cached keys are the previous window - 1 positions, all within reach
            if T != 1:
                raise ValueError("incremental local attention takes one byte at a time")
            k, v = torch.cat([cache["k"], k], dim=2), torch.cat([cache["v"], v], dim=2)
            mask = None
        if cache is not None:
            cache["k"], cache["v"] = k[:, :, -(self.window - 1) :], v[:, :, -(self.window - 1) :]
        a = F.scaled_dot_product_attention(q, k, v, attn_mask=mask, is_causal=mask is None and T > 1)
        h, c1 = self.canon1(x + self.proj(a.transpose(1, 2).reshape(B, T, C)), cache.get("c1") if cache else None)
        u = self.norm2(h)
        out, c2 = self.canon2(h + self.down(F.silu(self.gate(u)) * self.up(u)), cache.get("c2") if cache else None)
        if cache is not None:
            cache["c1"], cache["c2"] = c1, c2
        return out


class RateState:
    """Running inverse of M_t = I + (1/d) sum_{i<=t} n_i n_i^T for unit-RMS rows n_i, one per batch row.

    The marginal coding rate (equation 12 with epsilon^2 = d) is dR_t = 1/2 log(1 + (1/d) n_t^T M_{t-1}^{-1} n_t),
    by the matrix determinant lemma; the inverse is updated with the Sherman-Morrison formula."""

    def __init__(self, B: int, d: int, device):
        self.Minv = torch.eye(d, device=device).expand(B, d, d).clone()
        self.c = 1.0 / d

    def step(self, n: torch.Tensor) -> torch.Tensor:  # n (B, d) unit-RMS rows -> dR (B,)
        v = torch.bmm(self.Minv, n[:, :, None]).squeeze(-1)
        q = (n * v).sum(-1) * self.c
        self.Minv -= (self.c / (1 + q))[:, None, None] * v[:, :, None] * v[:, None, :]
        return 0.5 * torch.log1p(q)


@torch.no_grad()
def marginal_rate(h: torch.Tensor, mode: str, state: RateState | None = None) -> torch.Tensor:
    """Per-position marginal coding rate of the local representations (B, T) -> (B, T)."""
    hf = h.float()
    if mode == "l2":
        return hf.pow(2).sum(-1)
    n = hf / (hf.pow(2).mean(-1, keepdim=True).sqrt() + 1e-6)
    st = state or RateState(h.size(0), h.size(2), h.device)
    return torch.stack([st.step(n[:, t]) for t in range(h.size(1))], dim=1)


def select_topk(rates: torch.Tensor, K: int, valid: torch.Tensor | None = None) -> torch.Tensor:
    """Indices (B, K), sorted, of the K positions with the largest rates; position 0 is always included."""
    s = rates.clone()
    if valid is not None:
        s = s.masked_fill(~valid, float("-inf"))
    s[:, 0] = float("inf")
    return s.topk(K, dim=1).indices.sort(dim=1).values


class ByteFlowNet(nn.Module):
    def __init__(self, c: ByteConfig):
        super().__init__()
        self.c = c
        self.emb = nn.Embedding(VOCAB, c.d_local)
        self.encoder = nn.ModuleList(LocalBlock(c) for _ in range(c.n_local))
        self.proj = nn.Linear(c.d_local, c.d_global, bias=False)
        gc = GlobalConfig(vocab=VOCAB, d_model=c.d_global, n_layer=c.n_global, n_head=c.global_heads, d_ff=c.global_ff,
                          max_len=c.max_global, dropout=c.dropout)
        self.globals = nn.ModuleList(Block(gc) for _ in range(c.n_global))
        self.up_w = nn.Parameter(torch.randn(c.bins, c.d_global, c.d_local) * 0.02)
        self.decoder = nn.ModuleList(LocalBlock(c) for _ in range(c.n_local))
        self.norm = RMSNorm(c.d_local)
        self.out = nn.Linear(c.d_local, VOCAB, bias=False)
        lcos, lsin = rope_cache(c.d_local // c.local_heads, c.max_bytes)
        gcos, gsin = rope_cache(c.d_global // c.global_heads, c.max_global)
        for name, t in (("lcos", lcos), ("lsin", lsin), ("gcos", gcos), ("gsin", gsin)):
            self.register_buffer(name, t, persistent=False)
        for m in self.modules():
            if isinstance(m, (nn.Linear, nn.Embedding)):
                nn.init.normal_(m.weight, std=0.02)
        for blocks, n in ((self.encoder, c.n_local), (self.decoder, c.n_local), (self.globals, c.n_global)):
            for b in blocks:
                for lin in (b.proj, b.down):
                    nn.init.normal_(lin.weight, std=0.02 / math.sqrt(2 * n))

    def n_params(self) -> int:
        return sum(p.numel() for p in self.parameters())

    def k_for(self, T: int) -> int:
        return max(1, math.ceil(T / self.c.ratio))

    def encode(self, ids: torch.Tensor) -> torch.Tensor:
        x = self.emb(ids)
        for b in self.encoder:
            x = b(x, self.lcos, self.lsin)
        return x

    def global_forward(self, z: torch.Tensor) -> torch.Tensor:
        for b in self.globals:
            z = b(z, self.gcos, self.gsin)
        return z

    def upsample(self, h: torch.Tensor, g: torch.Tensor, idx: torch.Tensor) -> torch.Tensor:
        """s_t = h_t + g_{chunk(t)} W_{bin(t)}, bin = position within the chunk capped at bins - 1."""
        B, T, _ = h.shape
        sel = torch.zeros(B, T, dtype=torch.bool, device=h.device).scatter_(1, idx, True)
        cid = sel.long().cumsum(1) - 1
        start = idx.gather(1, cid)
        off = (torch.arange(T, device=h.device)[None, :] - start).clamp(max=self.c.bins - 1)
        gs = g.gather(1, cid[:, :, None].expand(-1, -1, g.size(2)))
        s = h
        for b in range(self.c.bins):
            m = (off == b)[:, :, None]
            if m.any():
                s = s + m * (gs @ self.up_w[b])
        return s

    def decode(self, s: torch.Tensor) -> torch.Tensor:
        for b in self.decoder:
            s = b(s, self.lcos, self.lsin)
        return self.out(self.norm(s))

    def forward(self, ids: torch.Tensor, labels: torch.Tensor | None = None, idx: torch.Tensor | None = None,
                valid: torch.Tensor | None = None):
        """Teacher-forced pass. `idx` fixes the chunk boundaries (else Top-K of the coding rate); `valid` masks padding."""
        h = self.encode(ids)
        if idx is None:
            idx = select_topk(marginal_rate(h, self.c.rate), self.k_for(ids.size(1)), valid)
        g = self.global_forward(self.proj(h.gather(1, idx[:, :, None].expand(-1, -1, h.size(2)))))
        logits = self.decode(self.upsample(h, g, idx))
        if labels is None:
            return logits, idx
        loss = F.cross_entropy(logits[:, :-1].reshape(-1, VOCAB).float(), labels[:, 1:].reshape(-1), ignore_index=-100)
        return logits, loss, idx

    # ----- generation (one sequence, with caches) -----
    @torch.no_grad()
    def generate(self, prompt: list[int], max_new: int = 400, no_repeat: int = 16, trace: dict | None = None,
                 max_chunk: int | None = None, rule: str = "prompt", window: int = 96) -> list[int]:
        """Greedy byte generation. `trace`, if given, receives the boundary positions and per-step logits.

        Boundary rules after the prompt (the paper defines Top-K for teacher forcing only):
        - "prompt": a byte starts a chunk when its marginal rate is at least the smallest rate in the prompt's Top-K.
          Marginal rates fall as context accumulates, so this rarely fires; kept for reference.
        - "quantile": a byte starts a chunk when its rate is at least the (1 - 1/ratio) quantile of the rates of the
          last `window` bytes, which keeps the training density of one boundary per `ratio` bytes, adaptively.
        `max_chunk` additionally forces a boundary once a chunk reaches that many bytes."""
        dev = self.emb.weight.device
        ids = torch.tensor([prompt], device=dev)
        T = ids.size(1)
        enc_c = [dict() for _ in self.encoder]
        dec_c = [dict() for _ in self.decoder]
        glob_c = [dict() for _ in self.globals]
        # prompt: encoder with caches, exact rates, Top-K selection, threshold for later boundaries
        x = self.emb(ids)
        for b, c in zip(self.encoder, enc_c):
            x = b(x, self.lcos, self.lsin, c)
        h = x
        state = RateState(1, self.c.d_local, dev)
        rates = marginal_rate(h, self.c.rate, state)
        idx = select_topk(rates, self.k_for(T))
        tau = rates[0, idx[0, 1:]].min().item() if idx.size(1) > 1 else float("inf")
        recent = rates[0, -window:].tolist()  # for the quantile rule
        q = 1.0 - 1.0 / self.c.ratio
        z = self.proj(h.gather(1, idx[:, :, None].expand(-1, -1, h.size(2))))
        K = z.size(1)
        for b, c in zip(self.globals, glob_c):
            z = b(z, self.gcos, self.gsin, torch.ones(1, 1, K, K, dtype=torch.bool, device=dev).tril(), c)
        g = z
        s = self.upsample(h, g, idx)
        for b, c in zip(self.decoder, dec_c):
            s = b(s, self.lcos, self.lsin, c)
        logits = self.out(self.norm(s[:, -1])).float()
        g_last, n_glob = g[:, -1:], K
        chunk_start = idx[0, -1].item()
        out: list[int] = []
        seen: dict[tuple[int, ...], set[int]] = {}
        pos = T
        if trace is not None:
            trace["prompt_boundaries"], trace["boundaries"], trace["logits"], trace["tau"] = idx[0].tolist(), [], [logits[0].clone()], tau
        for _ in range(max_new):
            if no_repeat and len(out) >= no_repeat - 1:
                key = tuple(out[len(out) - no_repeat + 1 :])
                if key in seen:
                    logits[0, list(seen[key])] = float("-inf")
            nxt = logits.argmax(-1).item()
            if nxt == EOS:
                break
            if no_repeat and len(out) >= no_repeat - 1:
                seen.setdefault(tuple(out[len(out) - no_repeat + 1 :]), set()).add(nxt)
            out.append(nxt)
            if pos >= self.c.max_bytes - 1:
                break
            # one incremental step for the new byte
            x = self.emb(torch.tensor([[nxt]], device=dev))
            for b, c in zip(self.encoder, enc_c):
                x = b(x, self.lcos[pos : pos + 1], self.lsin[pos : pos + 1], c)
            hn = x[:, 0].float()
            if self.c.rate == "l2":
                r = hn.pow(2).sum(-1)
            else:
                r = state.step(hn / (hn.pow(2).mean(-1, keepdim=True).sqrt() + 1e-6))
            forced = max_chunk is not None and pos - chunk_start >= max_chunk
            rv = r.item()
            if rule == "quantile":
                srt = sorted(recent)
                thr = srt[min(len(srt) - 1, int(q * len(srt)))]
                recent.append(rv)
                if len(recent) > window:
                    recent.pop(0)
                fire = rv >= thr
            else:
                fire = rv >= tau
            if (fire or forced) and n_glob < self.c.max_global:
                zc = self.proj(x)
                for b, c in zip(self.globals, glob_c):
                    zc = b(zc, self.gcos[n_glob : n_glob + 1], self.gsin[n_glob : n_glob + 1],
                           torch.ones(1, 1, 1, n_glob + 1, dtype=torch.bool, device=dev), c)
                g_last, n_glob, chunk_start = zc, n_glob + 1, pos
                if trace is not None:
                    trace["boundaries"].append(pos)
            off = min(pos - chunk_start, self.c.bins - 1)
            s = x + g_last @ self.up_w[off]
            for b, c in zip(self.decoder, dec_c):
                s = b(s, self.lcos[pos : pos + 1], self.lsin[pos : pos + 1], c)
            logits = self.out(self.norm(s[:, 0])).float()
            if trace is not None:
                trace["logits"].append(logits[0].clone())
            pos += 1
        return out
