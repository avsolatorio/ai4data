"""Evaluate current models on realistic searches and the query understanding against the generation spec.

For each narrated pair: the model's verdict from the first rounds' input format with only interface facets as filters
(v1, what a deployed model sees today) and with extracted constraints added as filters (v1x), against the teacher's
verdict on the new-format input. Breakdowns by search style, constraint placement and facets. Extractor recall is
measured against the spec elements the teacher reported as expressed.

  CUDA_VISIBLE_DEVICES=1 python -m relevance_explainer.synth.realistic_eval --data data/real_pilot --run runs/f6-dpo
"""
from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path

import torch
from tokenizers import Tokenizer

from ..tiny.model import Config, TinyLM
from ..tiny.scriptguard import banned_for, token_masks
from ..tiny.train import quantize

SEP = "\n\n### Judgment\n"
EOS = 1


def generate(run: Path, inputs: list[str], batch: int = 48) -> list[str]:
    ck = torch.load(run / "model_fp32.pt", map_location="cpu")
    model = TinyLM(Config(**ck["config"]))
    model.load_state_dict(ck["state"])
    model, _ = quantize(model.cuda().eval(), 8)
    tk = Tokenizer.from_file(str(run / "tokenizer.json"))
    masks = token_masks(tk)
    prompts = [tk.encode(t + SEP).ids[-(model.c.max_len - 270):] for t in inputs]
    order = sorted(range(len(prompts)), key=lambda i: len(prompts[i]))
    out = [""] * len(prompts)
    for s in range(0, len(order), batch):
        ch = order[s : s + batch]
        ban = torch.stack([banned_for(masks, inputs[i]) for i in ch])
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
            gens = model.generate_batch([prompts[i] for i in ch], EOS, 260, banned=ban)
        for i, g in zip(ch, gens):
            out[i] = tk.decode(g).strip()
    return out


def placement(r: dict) -> str:
    """Where the place and time constraints are: none, text only, facets only, or both."""
    p = r["parsed"]
    q = any(x["source"] == "query" for x in p["places"]) or (p.get("time") or {}).get("source") == "query"
    f = any(x["source"] == "facet" for x in p["places"]) or (p.get("time") or {}).get("source") == "facet"
    return "both" if q and f else "text only" if q else "facets only" if f else "none found"


def extractor_recall(rows: list[dict]) -> dict:
    stats = defaultdict(lambda: [0, 0])
    for r in rows:
        g, exp = r.get("geo"), set(r.get("expressed") or [])
        if g and "place" in exp:
            key = f"place:{g['kind']}"
            got = r["parsed"]["places"]
            if g["kind"] == "group":
                hit = any(x["kind"] == "group" and x.get("code") == g["code"] for x in got)
            elif g["kind"] == "global":
                hit = bool(r["parsed"].get("scope"))
            else:
                hit = all(any(x.get("iso3") == c for x in got) for c in g["iso3"])
            stats[key][0] += hit
            stats[key][1] += 1
            stats[f"place:{r['language']}"][0] += hit
            stats[f"place:{r['language']}"][1] += 1
        t = r.get("time")
        if t and "time" in exp:
            stats[f"time:{t['kind']}"][0] += bool(r["parsed"].get("time"))
            stats[f"time:{t['kind']}"][1] += 1
        if r.get("group") and "group" in exp:
            stats["group"][0] += any(x["value"] == r["group"] for x in r["parsed"]["groups"])
            stats["group"][1] += 1
    return {k: f"{v[0]}/{v[1]} ({100 * v[0] / v[1]:.0f}%)" for k, v in sorted(stats.items())}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--run", required=True)
    a = ap.parse_args()
    d, run = Path(a.data), Path(a.run)
    rows = [json.loads(l) for l in (d / "examples.jsonl").read_text().splitlines()]
    res = {}
    for fmt in ("v1", "v1x"):
        gens = generate(run, [r[f"input_{fmt}"] for r in rows])
        res[fmt] = [g.split("\n")[0].strip() for g in gens]
        with (d / f"gen_{run.name}_{fmt}.jsonl").open("w") as f:
            for r, g in zip(rows, gens):
                f.write(json.dumps({"id": r["id"], "generated": g}, ensure_ascii=False) + "\n")
    gold = [r["verdict"] for r in rows]
    acc = lambda pv, idx: sum(pv[i] == gold[i] for i in idx) / max(1, len(idx))  # noqa: E731
    allidx = list(range(len(rows)))
    print(f"examples {len(rows)}; teacher verdicts {dict(Counter(gold))}")
    print(f"{run.name}: v1 (facets only) {acc(res['v1'], allidx):.3f} | v1x (extracted constraints as filters) {acc(res['v1x'], allidx):.3f}")
    for name, key in (("placement", placement), ("style", lambda r: r["style"]), ("mode", lambda r: r["mode"]),
                      ("relation", lambda r: r.get("relation", "-")), ("source", lambda r: r["record"]["kind"]), ("language", lambda r: r["language"])):
        groups = defaultdict(list)
        for i, r in enumerate(rows):
            groups[key(r)].append(i)
        print(f"  by {name}: " + "; ".join(f"{k} n={len(v)} v1 {acc(res['v1'], v):.2f} v1x {acc(res['v1x'], v):.2f}" for k, v in sorted(groups.items())))
    print("extractor recall vs spec:", json.dumps(extractor_recall(rows), ensure_ascii=False))
    missed = Counter(m for r in rows for m in (r.get("missed") or []) if isinstance(m, str))
    print("teacher-flagged missed constraints:", sum(1 for r in rows if r.get("missed")), "examples; most common:", missed.most_common(12))


if __name__ == "__main__":
    main()
