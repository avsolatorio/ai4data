"""Low-rank adapters for the tiny model: per-language specialization on top of one shared base.

An adapter adds B @ A (rank r) to each linear layer of every block. The base stays frozen. `merge` folds the
adapters into the base weights, so inference and export are unchanged.
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from .model import TinyLM

TARGETS = ("qkv", "proj", "gate", "up", "down")


class LoRALinear(nn.Module):
    def __init__(self, base: nn.Linear, rank: int, alpha: float, dropout: float):
        super().__init__()
        self.base = base
        self.base.weight.requires_grad_(False)
        out_f, in_f = base.weight.shape
        dev = base.weight.device
        self.A = nn.Parameter(torch.empty(rank, in_f, device=dev))
        nn.init.kaiming_uniform_(self.A, a=math.sqrt(5))
        self.B = nn.Parameter(torch.zeros(out_f, rank, device=dev))  # zero start: the adapted model equals the base
        self.scale = alpha / rank
        self.drop = nn.Dropout(dropout)

    def forward(self, x):
        return self.base(x) + F.linear(F.linear(self.drop(x), self.A), self.B) * self.scale

    def merged(self) -> nn.Linear:
        lin = nn.Linear(self.base.in_features, self.base.out_features, bias=False)
        lin.weight = nn.Parameter(self.base.weight.detach() + (self.B.detach() @ self.A.detach()) * self.scale)
        return lin


def add_lora(model: TinyLM, rank: int = 16, alpha: float = 32.0, dropout: float = 0.0) -> list[nn.Parameter]:
    """Wrap the target linears of every block, freeze everything else, and return the trainable parameters."""
    for p in model.parameters():
        p.requires_grad_(False)
    params: list[nn.Parameter] = []
    for b in model.blocks:
        for name in TARGETS:
            wrapped = LoRALinear(getattr(b, name), rank, alpha, dropout)
            setattr(b, name, wrapped)
            params += [wrapped.A, wrapped.B]
    return params


def merge_lora(model: TinyLM) -> TinyLM:
    """Fold adapters into the base weights in place and drop the adapter modules."""
    for b in model.blocks:
        for name in TARGETS:
            m = getattr(b, name)
            if isinstance(m, LoRALinear):
                setattr(b, name, m.merged())
    for p in model.parameters():
        p.requires_grad_(True)
    return model


def lora_state(model: TinyLM) -> dict[str, torch.Tensor]:
    return {k: v.detach().cpu() for k, v in model.state_dict().items() if k.endswith(".A") or k.endswith(".B")}


def lora_bytes(state: dict[str, torch.Tensor], bits: int = 8) -> int:
    return sum(math.ceil(v.numel() * bits / 8) for v in state.values())
