"""Held-out bits per byte of a pretrained base under weight quantization schemes, to decide whether the embedding
table (also the output layer, since it is tied) can be stored at 4 bits.

Schemes: float; INT8 per row for every matrix (what tiny/export.py writes); INT8 matrices with the embedding at
4 bits in groups of G values with one fp16 scale per group; every matrix at 4 bits in groups of 32.
All schemes are scored on the same held-out sequences as the pretraining log (paired comparison).

  CUDA_VISIBLE_DEVICES=0 python scripts/quant_embedding_check.py runs/mb-v32k-d448-l7-gate data/corpus_multi_v32k [names]
"""
import copy
import json
import sys
from pathlib import Path

import torch
from tokenizers import Tokenizer

from relevance_explainer.tiny.model import Config, TinyLM
from relevance_explainer.tiny.pretrain import heldout_ppl, load_all_shards

run, root = Path(sys.argv[1]), Path(sys.argv[2])
EVAL_SEQS, HOLDOUT, SEQ = 8, 200_000, 1024


def q_rows(w: torch.Tensor, bits: int) -> tuple[torch.Tensor, int]:
    levels = 2 ** (bits - 1) - 1
    scale = w.abs().amax(dim=1, keepdim=True).clamp_min(1e-8) / levels
    return (w / scale).round().clamp(-levels, levels) * scale, w.numel() * bits // 8 + scale.numel() * 2


def q_groups(w: torch.Tensor, bits: int, g: int) -> tuple[torch.Tensor, int]:
    levels = 2 ** (bits - 1) - 1
    x = w.reshape(-1, g)
    scale = x.abs().amax(dim=1, keepdim=True).clamp_min(1e-8) / levels
    return ((x / scale).round().clamp(-levels, levels) * scale).reshape(w.shape), w.numel() * bits // 8 + scale.numel() * 2


def apply(model: TinyLM, scheme) -> tuple[TinyLM, int]:
    q, size = copy.deepcopy(model), 0
    for name, p in q.named_parameters():
        if p.dim() == 2:
            p.data, n = scheme(name, p.detach().float())
            size += n
        else:
            size += p.numel() * 2
    return q, size


ck = torch.load(run / "model.pt", map_location="cpu")
model = TinyLM(Config(**ck["config"]))
model.load_state_dict(ck["state"])
model = model.cuda()
tk = Tokenizer.from_file(str(root / "tokenizer.json"))
shards = load_all_shards(root)
emb_rows = model.emb.weight.numel()

schemes = {
    "float": None,
    "int8 all": lambda n, w: q_rows(w, 8),
    "int8 + emb int4 per row": lambda n, w: q_rows(w, 4) if n == "emb.weight" else q_rows(w, 8),
    "int8 + emb int4 g128": lambda n, w: q_groups(w, 4, 128) if n == "emb.weight" else q_rows(w, 8),
    "int8 + emb int4 g64": lambda n, w: q_groups(w, 4, 64) if n == "emb.weight" else q_rows(w, 8),
    "int8 + emb int4 g32": lambda n, w: q_groups(w, 4, 32) if n == "emb.weight" else q_rows(w, 8),
    "int4 g32 all": lambda n, w: q_groups(w, 4, 32),
    "int8 + emb int6 per row": lambda n, w: q_rows(w, 6) if n == "emb.weight" else q_rows(w, 8),
    "int8 + emb int6 g64": lambda n, w: q_groups(w, 6, 64) if n == "emb.weight" else q_rows(w, 8),
    "int8 + emb int5 g64": lambda n, w: q_groups(w, 5, 64) if n == "emb.weight" else q_rows(w, 8),
}
if len(sys.argv) > 3:  # a comma-separated subset of the schemes
    schemes = {k: v for k, v in schemes.items() if k in sys.argv[3].split(",")}
results = {}
for name, scheme in schemes.items():
    m, size = (model, sum(p.numel() * 4 for p in model.parameters())) if scheme is None else apply(model, scheme)
    ppl = heldout_ppl(m, shards, SEQ, n_seq=EVAL_SEQS, tk=tk, holdout=HOLDOUT)
    bpb = {k[:-4]: v for k, v in ppl.items() if k.endswith("_bpb")}
    results[name] = {"mb": size / 1e6, "bpb": bpb}
    print(f"{name:26s} {size / 1e6:6.1f} MB  mean bits per byte {sum(bpb.values()) / len(bpb):.4f}", flush=True)
path = run / "quant_check.json"
path.write_text(json.dumps({**(json.loads(path.read_text()) if path.exists() else {}), **results}, indent=1))
