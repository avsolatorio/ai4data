"""Bits per byte of a base model by position in each language's shard: the first 8 held-out sequences (what training
evaluated), the rest of the held-out tail, and windows of training text at the start, middle and just before the
tail. Tests whether text difficulty changes along the shard, which would confound comparisons between positions.

  python scripts/eval_base_position.py runs/base-final/model.pt data/corpus_multi_final
"""
import math
import sys
from pathlib import Path

import numpy as np
import torch
from tokenizers import Tokenizer

from relevance_explainer.tiny.model import Config, TinyLM
from relevance_explainer.tiny.pretrain import load_all_shards



@torch.no_grad()
def bpb(model, tk, toks: np.ndarray, seq: int) -> float:
    """Bits per UTF-8 byte over whole sequences of `toks` (as scripts/eval_base.py)."""
    n = len(toks) // seq
    x = torch.from_numpy(toks[: n * seq].astype(np.int64).reshape(n, seq)).cuda()
    tot = 0.0
    for i in range(0, n, 32):
        with torch.autocast("cuda", dtype=torch.bfloat16):
            _, loss = model(x[i : i + 32], x[i : i + 32])
        tot += loss.item() * len(x[i : i + 32])
    return tot * (seq - 1) / math.log(2) / len(tk.decode(toks[: n * seq].tolist()).encode("utf-8"))


ck = torch.load(sys.argv[1], map_location="cpu", weights_only=False)
model = TinyLM(Config(**ck["config"])).cuda().eval()
model.load_state_dict(ck["state"])
seq, H = 1024, 200000  # the pretraining sequence length (max_len is larger; positions past 1024 were never trained)
root = Path(sys.argv[2])
tk = Tokenizer.from_file(str(root / "tokenizer.json"))
shards = load_all_shards(root)
cols = {"held-out first 8": [], "held-out rest": [], "train start": [], "train middle": [], "train before tail": []}
for l, s in shards.items():
    s = np.asarray(s)
    n, w = len(s), 32 * seq
    tail = s[-H:]
    cols["held-out first 8"].append(bpb(model, tk, tail[: 8 * seq], seq))
    cols["held-out rest"].append(bpb(model, tk, tail[8 * seq : 195 * seq], seq))
    cols["train start"].append(bpb(model, tk, s[:w], seq))
    m = (n - H) // 2
    cols["train middle"].append(bpb(model, tk, s[m : m + w], seq))
    cols["train before tail"].append(bpb(model, tk, s[n - H - w : n - H], seq))
import json  # noqa: E402
plan = json.loads((root / "plan.json").read_text())
langs = list(shards)
groups = {"all": range(len(langs)), "4 passes": [i for i, l in enumerate(langs) if plan.get(l, {}).get("passes", 0) >= 3.5],
          "under 1 pass": [i for i, l in enumerate(langs) if plan.get(l, {}).get("passes", 0) < 1]}
for g, ix in groups.items():
    print(g, len(ix), "languages: " + ", ".join(f"{k} {sum(v[i] for i in ix) / len(ix):.4f}" for k, v in cols.items()))
