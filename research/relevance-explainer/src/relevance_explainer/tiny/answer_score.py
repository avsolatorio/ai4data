"""Answer confidence: the probability that an answer is fully correct (verdict matches the teacher and the explanation is
judged good), from the uncertainty signals the model computes while generating. See tiny/selfcheck.py for the features
and scripts/save_score_head.py for the 9-parameter head.

The decoder records, at each step, the log-probability of the chosen token, the entropy and the end-of-text
probability of the unmasked distribution (`step_stats`); `features` turns those into the head's inputs exactly as
`selfcheck.output_features` does from a teacher-forced pass, so scoring needs no extra forward pass.
"""
from __future__ import annotations

import json
import math
from pathlib import Path

import torch

EOS = 1


def step_stats(logits: torch.Tensor, chosen: int) -> tuple[float, float, float]:
    """(log-probability of the chosen token, entropy, end-of-text probability) of one step's unmasked logits."""
    lp = torch.log_softmax(logits.float(), -1)
    return lp[chosen].item(), -(lp.exp() * lp).sum().item(), lp[EOS].exp().item()


def features(steps: list[tuple[float, float, float]], out: list[int], newline: int, verdict_prob: float,
             invented: bool, prefix: int = 0) -> list[float]:
    """Head inputs for one output.

    `out` are the output tokens (without end of text). `steps[k]` are the stats of the step that produced token
    `prefix + k`; a step after the last token (the one that produced end of text) may follow. `prefix` counts output
    tokens that were given rather than generated (a forced verdict line); they must end with the newline, so every
    explanation token has a step."""
    gen = len(out) - prefix
    lps = [s[0] for s in steps[:gen]]
    ents = [s[1] for s in steps[:gen]]
    nl = out.index(newline) if newline in out else 0
    skip = max(0, nl + 1 - prefix)  # explanation tokens start after the verdict line
    narr_lp, narr_ent = lps[skip:], ents[skip:]
    if not narr_lp:
        narr_lp, narr_ent = lps, ents
    if not narr_lp:  # empty output
        narr_lp, narr_ent = [0.0], [0.0]
    p_eos = steps[gen][2] if len(steps) > gen else steps[-1][2] if steps else 0.0
    return [verdict_prob, sum(narr_lp) / len(narr_lp), min(narr_lp), sum(narr_ent) / len(narr_ent), max(narr_ent),
            math.log1p(len(out)), p_eos, float(invented)]


class ScoreHead:
    def __init__(self, path: str | Path):
        d = json.loads(Path(path).read_text())
        self.mean, self.std, self.weight, self.bias = d["mean"], d["std"], d["weight"], d["bias"]

    def __call__(self, feats: list[float]) -> float:
        z = self.bias + sum(w * (f - m) / s for w, f, m, s in zip(self.weight, feats, self.mean, self.std))
        return 1 / (1 + math.exp(-z))
