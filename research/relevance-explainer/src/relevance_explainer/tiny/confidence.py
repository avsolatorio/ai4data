"""Verdict confidence: the model's probability of each verdict line given the input, and how accuracy varies with it.

The three verdict lines ("relevant\\n", "partly_relevant\\n", "irrelevant\\n") are scored by teacher forcing after the
prompt; normalised over the three, they give a distribution whose maximum is the confidence of the greedy verdict.
A catalog can show low-confidence verdicts as uncertain.

  python -m relevance_explainer.tiny.confidence --model runs/v2f-nosim-dpo/explainer-int8.pt \\
      --data data/eval_real/itemfirst_nosim_v3 --out runs/v2f-nosim-dpo/realtest_v3_confidence.jsonl
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
import torch.nn.functional as F

from .export import load_int8

SEP = "\n\n### Judgment\n"
VERDICTS = ("relevant", "partly_relevant", "irrelevant")


@torch.no_grad()
def verdict_probs(model, tk, inputs: list[str]) -> list[dict]:
    """Probability of each verdict for each rendered input (normalised over the three verdicts)."""
    dev = model.emb.weight.device
    out = []
    for text in inputs:
        prompt = tk.encode(text + SEP).ids
        seqs = []
        for v in VERDICTS:
            full = tk.encode(text + SEP + v + "\n").ids
            assert full[: len(prompt)] == prompt, "verdict tokens merge with the separator"
            seqs.append(full)
        lp = []
        for full in seqs:
            ids = torch.tensor([full[-model.c.max_len:]], device=dev)
            n_tgt = len(full) - len(prompt)
            start = ids.size(1) - n_tgt
            prefix = torch.tensor([start], device=dev) if model.c.prefix_lm else None
            logp = F.log_softmax(model(ids, prefix=prefix)[0].float(), -1)
            lp.append(sum(logp[start - 1 + k, ids[0, start + k]].item() for k in range(n_tgt)))
        p = torch.tensor(lp).softmax(0).tolist()
        out.append(dict(zip(VERDICTS, p)))
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--data", required=True, help="evaluation directory with test.jsonl")
    ap.add_argument("--out", required=True)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    a = ap.parse_args()
    model, tk = load_int8(a.model, a.device)
    rows = [json.loads(l) for l in (Path(a.data) / "test.jsonl").read_text().splitlines()]
    probs = verdict_probs(model, tk, [r["input"] for r in rows])
    res = []
    for r, p in zip(rows, probs):
        pred = max(p, key=p.get)
        res.append({"probs": p, "pred": pred, "conf": p[pred], "label": r["verdict"], "unanimous": r.get("unanimous"),
                    "relation": r.get("relation"), "source": r.get("source")})
    Path(a.out).write_text("".join(json.dumps(x) + "\n" for x in res))
    res.sort(key=lambda x: -x["conf"])
    n = len(res)
    print(f"accuracy of the most probable verdict: {sum(x['pred'] == x['label'] for x in res) / n:.3f} (n={n})")
    print("coverage  min-conf  accuracy   accuracy-on-unanimous")
    for cov in (1.0, 0.9, 0.8, 0.7, 0.6, 0.5):
        k = max(1, int(n * cov))
        top = res[:k]
        u = [x for x in top if x["unanimous"]]
        print(f"{cov:8.0%}  {top[-1]['conf']:8.2f}  {sum(x['pred'] == x['label'] for x in top) / k:8.3f}   "
              f"{sum(x['pred'] == x['label'] for x in u) / max(1, len(u)):.3f}")
    print("calibration: confidence bin -> accuracy (n)")
    for lo, hi in ((0.0, 0.5), (0.5, 0.7), (0.7, 0.9), (0.9, 0.97), (0.97, 1.01)):
        b = [x for x in res if lo <= x["conf"] < hi]
        if b:
            print(f"  {lo:.2f}-{min(hi, 1):.2f}: {sum(x['pred'] == x['label'] for x in b) / len(b):.3f} ({len(b)})")


if __name__ == "__main__":
    main()
