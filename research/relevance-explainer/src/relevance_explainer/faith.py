"""Features and scoring for the live faithfulness scorer (scripts/faith_scorer.py fits it; the fitted model is
<run>/faith_scorer.json). The score is the probability that the teacher judge would rate the explanation of a format-C
answer faithful 4 or 5, from the verifier's checks and the generating model's own view of its answer."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch

from .claims import from_model, plain_output, verify


def verifier_features(answer: str, inp: str) -> list[float]:
    """answer: the full tagged output, verdict line first."""
    ck = verify(from_model(answer), inp)
    return [float(ck.all_verified), ck.verified_share if ck.verified_share is not None else 1.0,
            float(len(ck.unverified)), float(len(ck.errors)), float(len(ck.refs)),
            float(len(plain_output(answer.partition("\n")[2]).split()))]


@torch.no_grad()
def model_features(model, prompts: list[list[int]], answers: list[list[int]], batch: int = 32,
                   max_len: int = 2048) -> tuple[np.ndarray, np.ndarray]:
    """Token ids of the prompts (up to and including the verdict line) and of the answers after it (EOS included).
    Returns the log-probability features [n, 6] (NaN when the answer does not fit) and the final hidden states [n, 2d]
    (mean over the answer positions, then the last position)."""
    store = {}
    hook = model.norm.register_forward_hook(lambda m, i, o: store.__setitem__("h", o))
    gen, hid = [None] * len(prompts), [None] * len(prompts)
    order = sorted(range(len(prompts)), key=lambda j: len(prompts[j]) + len(answers[j]))
    try:
        for b in range(0, len(order), batch):
            ch = order[b : b + batch]
            seqs = [(prompts[j] + answers[j])[:max_len] for j in ch]
            ids = torch.zeros(len(ch), max(map(len, seqs)), dtype=torch.long)
            for r, s in enumerate(seqs):
                ids[r, : len(s)] = torch.tensor(s)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                logits = model(ids.cuda())
            lp = torch.log_softmax(logits.float(), -1)
            h = store["h"].float()
            for r, j in enumerate(ch):
                p0, n = len(prompts[j]), len(seqs[r])
                if p0 >= n:
                    gen[j], hid[j] = np.full(6, np.nan), np.zeros(2 * h.size(-1), np.float32)
                    continue
                l = lp[r, p0 - 1 : n - 1]
                tok = l.gather(1, ids[r, p0:n].cuda()[:, None])[:, 0]
                ent = -(l.exp() * l).sum(-1)
                hv = h[r, p0 - 1 : n - 1]
                gen[j] = np.array([tok.mean().item(), tok.min().item(), torch.quantile(tok, 0.05).item(),
                                   (tok < np.log(0.1)).float().mean().item(), ent.mean().item(), float(n - p0)])
                hid[j] = torch.cat([hv.mean(0), hv[-1]]).cpu().numpy().astype(np.float32)
    finally:
        hook.remove()
    return np.stack(gen), np.stack(hid)


class FaithScorer:
    def __init__(self, path: str | Path):
        d = json.loads(Path(path).read_text())
        self.mean, self.scale = np.array(d["mean"]), np.array(d["scale"])
        self.coef, self.intercept = np.array(d["coef"]), d["intercept"]

    def proba(self, ver: np.ndarray, gen: np.ndarray, hid: np.ndarray) -> np.ndarray:
        x = (np.hstack([ver, gen, hid]) - self.mean) / self.scale
        return 1 / (1 + np.exp(-(x @ self.coef + self.intercept)))
