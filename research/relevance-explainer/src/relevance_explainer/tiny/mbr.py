"""Minimum Bayes risk (consensus) decoding of the narrative.

The verdict comes from greedy decoding. With that verdict as a fixed prefix, the model samples N narratives; the
greedy narrative is added as a candidate, and the candidate with the highest mean chrF against the others is kept.
The verdict is therefore unchanged and only the narrative can differ.

Only the examples the judge will see (the usual seeded sample of the test generations) are decoded this way; the
other rows are copied from the greedy file so that the judge's sample is the same.

  CUDA_VISIBLE_DEVICES=1 python -m relevance_explainer.tiny.mbr --run runs/f6-dpo --greedy-tag f2testsg --tag f2testmbr
"""
from __future__ import annotations

import argparse
import json
import random
from collections import Counter
from pathlib import Path

import torch
from tokenizers import Tokenizer

from .model import Config, TinyLM
from .scriptguard import banned_for, token_masks
from .train import quantize

SEP = "\n\n### Judgment\n"
EOS = 1


def _ngrams(s: str, n: int) -> Counter:
    s = " ".join(s.split())
    return Counter(s[i : i + n] for i in range(len(s) - n + 1))


def chrf(hyp: str, ref: str, n_max: int = 6, beta: float = 2.0) -> float:
    """Character n-gram F-score (chrF), averaged over n = 1..n_max."""
    ps, rs = [], []
    for n in range(1, n_max + 1):
        h, r = _ngrams(hyp, n), _ngrams(ref, n)
        if not h or not r:
            continue
        m = sum((h & r).values())
        ps.append(m / sum(h.values()))
        rs.append(m / sum(r.values()))
    if not ps:
        return 0.0
    p, r = sum(ps) / len(ps), sum(rs) / len(rs)
    return 0.0 if p + r == 0 else (1 + beta**2) * p * r / (beta**2 * p + r)


def select(cands: list[str]) -> int:
    """Index of the candidate with the highest mean chrF against all the other candidates."""
    if len(cands) == 1:
        return 0
    scores = [sum(chrf(c, o) for j, o in enumerate(cands) if j != i) / (len(cands) - 1) for i, c in enumerate(cands)]
    return max(range(len(cands)), key=scores.__getitem__)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--greedy-tag", default="f2testsg")
    ap.add_argument("--tag", default="f2testmbr")
    ap.add_argument("--prec", default="int8")
    ap.add_argument("--n", type=int, default=800, help="judge sample size")
    ap.add_argument("--samples", type=int, default=8)
    ap.add_argument("--temp", type=float, default=0.7)
    ap.add_argument("--top-k", type=int, default=40)
    ap.add_argument("--max-new", type=int, default=240)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()

    run = Path(a.run)
    rows = [json.loads(l) for l in (run / f"{a.greedy_tag}_generations_{a.prec}.jsonl").read_text().splitlines()]
    pick = random.Random(0).sample(range(len(rows)), min(a.n, len(rows)))  # same draw as judge.py on these rows
    ck = torch.load(run / "model_fp32.pt", map_location="cpu")
    model = TinyLM(Config(**ck["config"]))
    model.load_state_dict(ck["state"])
    model = model.cuda().eval()
    if a.prec != "fp32":
        model, _ = quantize(model, {"int8": 8, "int4": 4}[a.prec])
    tk = Tokenizer.from_file(str(run / "tokenizer.json"))
    masks = token_masks(tk)
    torch.manual_seed(a.seed)

    jobs = []
    for i in pick:
        verdict, _, narrative = rows[i]["generated"].strip().partition("\n")
        prompt = tk.encode(rows[i]["input"] + SEP + verdict.strip() + "\n").ids
        jobs += [(i, prompt)] * a.samples
    samples: dict[int, list[str]] = {i: [] for i in pick}
    order = sorted(range(len(jobs)), key=lambda k: len(jobs[k][1]))
    for s in range(0, len(order), a.batch):
        ch = order[s : s + a.batch]
        ban = torch.stack([banned_for(masks, rows[jobs[k][0]]["input"]) for k in ch])
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
            outs = model.generate_batch([jobs[k][1] for k in ch], EOS, a.max_new, temperature=a.temp, top_k=a.top_k, banned=ban)
        for k, o in zip(ch, outs):
            if len(o) < a.max_new:  # drop truncated samples
                samples[jobs[k][0]].append(tk.decode(o).strip())

    changed = 0
    out = [dict(r) for r in rows]
    for i in pick:
        verdict, _, greedy = rows[i]["generated"].strip().partition("\n")
        cands = [greedy.strip()] + [c for c in samples[i] if c]
        best = cands[select(cands)]
        changed += best != greedy.strip()
        out[i]["generated"] = verdict.strip() + "\n" + best
    with (run / f"{a.tag}_generations_{a.prec}.jsonl").open("w") as f:
        for r in out:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(json.dumps({"examples": len(pick), "samples": a.samples, "temp": a.temp, "replaced_greedy": changed}))


if __name__ == "__main__":
    main()
