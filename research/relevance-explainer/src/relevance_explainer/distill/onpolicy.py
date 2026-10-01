"""On-policy data for preference training: the student writes narratives, the teacher scores them.

  sample   student samples K narratives per training input, with the teacher's verdict as a forced prefix
  score    teacher judges every sample (faithfulness, coherence, verdict consistency, language)
  pairs    build (chosen, rejected) pairs for the same input from the scores

Servers are given as TEACHER_URLS (see teacher_checks).
"""
from __future__ import annotations

import argparse
import json
import random
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import torch
from tokenizers import Tokenizer

from ..tiny.judge import SYSTEM as JUDGE_SYSTEM
from ..tiny.model import Config, TinyLM
from ..synth.llm import parse_json
from .teacher_checks import MODEL, post, urls

SEP = "\n\n### Judgment\n"
EOS = 1


def _read(p: Path) -> list[dict]:
    return [json.loads(l) for l in p.read_text().splitlines()]


def cmd_sample(a) -> None:
    run, data = Path(a.run), Path(a.data)
    ck = torch.load(run / "model_fp32.pt", map_location="cpu")
    model = TinyLM(Config(**ck["config"]))
    model.load_state_dict(ck["state"])
    model = model.cuda().eval()
    tk = Tokenizer.from_file(str(run / "tokenizer.json"))
    rows = _read(data / "train.jsonl")
    idx = random.Random(a.seed).sample(range(len(rows)), min(a.n, len(rows)))
    out = data / a.out
    done = {d["i"] for d in _read(out)} if out.exists() else set()
    idx = [i for i in idx if i not in done]

    torch.manual_seed(a.seed)
    jobs = [(i, k) for i in idx for k in range(a.k)]
    prompts = {i: tk.encode(rows[i]["input"] + SEP + rows[i]["verdict"] + "\n").ids for i in idx}
    order = sorted(jobs, key=lambda j: len(prompts[j[0]]))
    samples: dict[int, list[str]] = {i: [] for i in idx}
    for s in range(0, len(order), a.batch):
        ch = order[s : s + a.batch]
        with torch.autocast("cuda", dtype=torch.bfloat16):
            outs = model.generate_batch([prompts[i] for i, _ in ch], EOS, a.max_new, temperature=a.temp, top_k=40)
        for (i, _), o in zip(ch, outs):
            samples[i].append(tk.decode(o).strip())
    with out.open("a") as f:
        for i in idx:
            f.write(json.dumps({"i": i, "verdict": rows[i]["verdict"], "samples": samples[i]}, ensure_ascii=False) + "\n")
    print(f"sampled {len(idx)} inputs x {a.k}")


def score_one(base: str, inp: str, verdict: str, narrative: str) -> dict | None:
    body = {"model": MODEL, "max_tokens": 100, "temperature": 0.0,
            "messages": [{"role": "system", "content": JUDGE_SYSTEM},
                         {"role": "user", "content": f"INPUT:\n{inp}\n\nEXPLANATION:\n{verdict}\n{narrative}"}],
            "chat_template_kwargs": {"enable_thinking": False}}
    try:
        j = parse_json(post(base, body)["choices"][0]["message"]["content"])
        return {"faithful": int(j["faithful"]), "coherent": int(j["coherent"]),
                "consistent": bool(j["consistent"]), "language_ok": bool(j["language_ok"])}
    except Exception:
        return None


def cmd_score(a) -> None:
    data = Path(a.data)
    rows = _read(data / "train.jsonl")
    samples = _read(data / a.samples)
    out = data / a.out
    done = {d["i"] for d in _read(out)} if out.exists() else set()
    todo = [d for d in samples if d["i"] not in done]
    servers = urls()
    jobs = [(d["i"], k, t) for d in todo for k, t in enumerate(d["samples"])]
    with ThreadPoolExecutor(a.workers) as ex:
        res = list(ex.map(lambda j: score_one(servers[(j[0] + j[1]) % len(servers)], rows[j[0]]["input"], rows[j[0]]["verdict"], j[2]), jobs))
    by: dict[int, list] = {d["i"]: [None] * len(d["samples"]) for d in todo}
    for (i, k, _), s in zip(jobs, res):
        by[i][k] = s
    with out.open("a") as f:
        for d in todo:
            f.write(json.dumps({"i": d["i"], "verdict": d["verdict"], "samples": d["samples"], "scores": by[d["i"]]}, ensure_ascii=False) + "\n")
    print(f"scored {len(todo)} inputs ({sum(s is None for v in by.values() for s in v)} judge failures)")


def quality(s: dict) -> float:
    """Single number for ranking a scored sample; consistency and language are required, not traded off."""
    return s["faithful"] * 2 + s["coherent"] - (0 if s["consistent"] and s["language_ok"] else 10)


def cmd_pairs(a) -> None:
    data = Path(a.data)
    rows = _read(data / "train.jsonl")
    pairs, stats = [], {"no_pair": 0, "chosen_from_student": 0, "chosen_from_teacher": 0}
    for d in _read(data / a.scored):
        cand = [(t, s) for t, s in zip(d["samples"], d["scores"]) if s and t]
        if len(cand) < 2:
            stats["no_pair"] += 1
            continue
        cand.sort(key=lambda c: quality(c[1]))
        worst, best = cand[0], cand[-1]
        teacher = rows[d["i"]]["output"].split("\n", 1)[1]
        if worst[1]["faithful"] > a.max_rejected_faithful and worst[1]["coherent"] > 2:
            stats["no_pair"] += 1  # nothing clearly bad to push away from
            continue
        if best[1]["faithful"] >= 5 and best[1]["coherent"] >= 4 and best[1]["consistent"] and best[1]["language_ok"]:
            chosen = best[0]
            stats["chosen_from_student"] += 1
        else:
            chosen = teacher
            stats["chosen_from_teacher"] += 1
        pairs.append({"i": d["i"], "verdict": d["verdict"], "chosen": chosen, "rejected": worst[0]})
    with (data / a.out).open("w") as f:
        for p in pairs:
            f.write(json.dumps(p, ensure_ascii=False) + "\n")
    print(f"{len(pairs)} pairs", stats)


def main() -> None:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("sample")
    p.add_argument("--run", required=True); p.add_argument("--data", required=True)
    p.add_argument("--n", type=int, default=6000); p.add_argument("--k", type=int, default=4)
    p.add_argument("--temp", type=float, default=0.8); p.add_argument("--batch", type=int, default=96)
    p.add_argument("--max-new", type=int, default=200); p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", default="onpolicy_samples.jsonl"); p.set_defaults(fn=cmd_sample)
    p = sub.add_parser("score")
    p.add_argument("--data", required=True); p.add_argument("--samples", default="onpolicy_samples.jsonl")
    p.add_argument("--out", default="onpolicy_scored.jsonl"); p.add_argument("--workers", type=int, default=256)
    p.set_defaults(fn=cmd_score)
    p = sub.add_parser("pairs")
    p.add_argument("--data", required=True); p.add_argument("--scored", default="onpolicy_scored.jsonl")
    p.add_argument("--out", default="dpo_pairs.jsonl"); p.add_argument("--max-rejected-faithful", type=int, default=3)
    p.set_defaults(fn=cmd_pairs)
    a = ap.parse_args()
    a.fn(a)


if __name__ == "__main__":
    main()
