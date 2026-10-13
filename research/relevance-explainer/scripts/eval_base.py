"""Full held-out evaluation of a pretrained base: bits per byte per language on every held-out sequence (the last
--holdout tokens of each language, as in training), and the same measure on text the model trained on (the start of
each shard). A training-text score far below the held-out score in languages repeated up to 4 times would mean the
model memorized them. Writes <out>.json and prints group means.

  python scripts/eval_base.py runs/base-final/model.pt data/corpus_multi_final runs/base-final/eval_full [--seqs 195]
"""
import argparse
import json
import math
from pathlib import Path

import numpy as np
import torch
from tokenizers import Tokenizer

from relevance_explainer.tiny.model import Config, TinyLM
from relevance_explainer.tiny.pretrain import load_all_shards

OURS = {"eng_Latn", "fra_Latn", "spa_Latn", "por_Latn", "arb_Arab", "cmn_Hani", "rus_Cyrl", "ind_Latn", "hin_Deva",
        "tur_Latn", "swh_Latn"}


@torch.no_grad()
def bpb(model, tk, toks: np.ndarray, seq: int) -> float:
    n = len(toks) // seq
    x = torch.from_numpy(toks[: n * seq].astype(np.int64).reshape(n, seq)).cuda()
    tot = 0.0
    for i in range(0, n, 32):
        with torch.autocast("cuda", dtype=torch.bfloat16):
            _, loss = model(x[i : i + 32], x[i : i + 32])
        tot += loss.item() * len(x[i : i + 32])
    nbytes = len(tk.decode(toks[: n * seq].tolist()).encode("utf-8"))
    return tot * (seq - 1) / math.log(2) / nbytes


ap = argparse.ArgumentParser()
ap.add_argument("ckpt"); ap.add_argument("root"); ap.add_argument("out")
ap.add_argument("--seqs", type=int, default=195, help="held-out sequences per language (200k tokens hold 195)")
ap.add_argument("--train-seqs", type=int, default=32, help="sequences of training text per language")
ap.add_argument("--holdout", type=int, default=200000)
ap.add_argument("--seq", type=int, default=1024, help="sequence length (at most the length the model was trained on)")
a = ap.parse_args()

root = Path(a.root)
ck = torch.load(a.ckpt, map_location="cpu", weights_only=False)
model = TinyLM(Config(**ck["config"])).cuda().eval()
model.load_state_dict(ck["state"])
seq = a.seq  # the pretraining sequence length; positions past it were never trained
tk = Tokenizer.from_file(str(root / "tokenizer.json"))
plan = json.loads((root / "plan.json").read_text())
shards = load_all_shards(root)
res = {}
for l, s in shards.items():
    tail = np.asarray(s[-a.holdout:])[: a.seqs * seq]
    head = np.asarray(s[: a.train_seqs * seq])
    res[l] = {"heldout": round(bpb(model, tk, tail, seq), 4), "train": round(bpb(model, tk, head, seq), 4),
              "passes": plan.get(l, {}).get("passes")}
Path(a.out + ".json").write_text(json.dumps({"ckpt": a.ckpt, "seqs": a.seqs, "langs": res}, indent=1))
small = set(sorted(plan, key=lambda k: plan[k]["chars"])[: len(plan) // 3])
groups = {"all": list(res), "our 11": [l for l in res if l in OURS], "smallest third": [l for l in res if l in small],
          "4 passes": [l for l in res if (res[l]["passes"] or 0) >= 3.5],
          "under 1 pass": [l for l in res if (res[l]["passes"] or 0) < 1]}
for g, ls in groups.items():
    h = sum(res[l]["heldout"] for l in ls) / len(ls)
    t = sum(res[l]["train"] for l in ls) / len(ls)
    print(f"{g:15s} {len(ls):3d} languages  held-out {h:.4f}  training text {t:.4f}  gap {h - t:+.4f}")
