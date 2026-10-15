"""On-policy preference pairs for the claim-carrying format (C): the student samples answers after the teacher's
verdict, the verifier checks every sample, the teacher judges only the candidates for a pair, and pairs keep the
verdict fixed (it is in the prompt), so the preference acts on the grade, claims, tl;dr and explanation only.

  sample   K answers per training input (tags kept, no repeat ban), verdict as a forced prefix
  score    verifier on every sample (claims.verify), coverage and length against the teacher's own answer; the teacher
           judges the explanation (words only) of the 2 best and the worst sample by the verifier
  pairs    chosen: a judged sample that is fully verified, covers the facts and is judged faithful 5 and coherent 4+
           (else the teacher's answer); rejected: the worst judged sample if it is clearly bad (a failed reference or
           unverified mention, missing coverage, or faithful 3 or lower)

Coverage: the sample references at least as many distinct claims as the teacher's answer minus one (at least one
when the teacher has any), so a sample cannot rank well by saying less. Length: the explanation has at least 60% of
the teacher's words. Servers are given as TEACHER_URLS (see teacher_checks).
"""
from __future__ import annotations

import argparse
import json
import random
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import torch
from tokenizers import Tokenizer

from ..claims import NO_REPEAT_TAGGED, from_model, plain_output, verify
from ..tiny.model import Config, TinyLM
from .onpolicy import SEP, EOS, _read, score_one
from .teacher_checks import urls


def _words(text: str) -> int:
    return len(plain_output(text).partition("\n")[2].split())


def check(full: str, inp: str, teacher_full: str) -> dict:
    ck, tk = verify(from_model(full), inp), verify(from_model(teacher_full), inp)
    n_s, n_t = len({r.claim for r in ck.refs}), len({r.claim for r in tk.refs})
    return {"verified": ck.all_verified, "share": ck.verified_share if ck.verified_share is not None else 1.0,
            "unverified": len(ck.unverified), "errors": len(ck.errors), "refs": len(ck.refs),
            "coverage": n_s >= (max(1, n_t - 1) if n_t else 0),
            "length": _words(full) >= 0.6 * _words(teacher_full)}


def rank(c: dict) -> tuple:
    return (c["verified"] and c["coverage"] and c["length"] and not c["errors"], c["share"], -c["unverified"])


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
    prompts = {i: tk.encode(rows[i]["input"] + SEP + rows[i]["verdict"] + "\n").ids for i in idx}
    jobs = sorted(((i, k) for i in idx for k in range(a.k)), key=lambda j: len(prompts[j[0]]))
    samples: dict[int, list[str]] = {i: [] for i in idx}
    for s in range(0, len(jobs), a.batch):
        ch = jobs[s : s + a.batch]
        with torch.autocast("cuda", dtype=torch.bfloat16):
            outs = model.generate_batch([prompts[i] for i, _ in ch], EOS, a.max_new, temperature=a.temp, top_k=40,
                                        no_repeat=NO_REPEAT_TAGGED)
        for (i, _), o in zip(ch, outs):
            samples[i].append(tk.decode(o, skip_special_tokens=False).strip())
    with out.open("a") as f:
        for i in idx:
            f.write(json.dumps({"i": i, "verdict": rows[i]["verdict"], "samples": samples[i]}, ensure_ascii=False) + "\n")
    print(f"sampled {len(idx)} inputs x {a.k}", flush=True)


def cmd_score(a) -> None:
    data = Path(a.data)
    rows = _read(data / "train.jsonl")
    samples = _read(data / a.samples)
    out = data / a.out
    done = {d["i"] for d in _read(out)} if out.exists() else set()
    todo = [d for d in samples if d["i"] not in done]
    servers = urls()
    recs, jobs = [], []
    for d in todo:
        r = rows[d["i"]]
        checks = [check(d["verdict"] + "\n" + s, r["input"], r["output"]) for s in d["samples"]]
        order = sorted(range(len(checks)), key=lambda k: rank(checks[k]), reverse=True)
        judge_k = sorted(set(order[:2] + order[-1:]))
        recs.append((d, checks, judge_k))
        jobs += [(d["i"], k, plain_output(d["verdict"] + "\n" + d["samples"][k]).partition("\n")[2]) for k in judge_k]
    with ThreadPoolExecutor(a.workers) as ex:
        res = list(ex.map(lambda j: score_one(servers[(j[0] + j[1]) % len(servers)], rows[j[0]]["input"],
                                              rows[j[0]]["verdict"], j[2]), jobs))
    judged = {(i, k): s for (i, k, _), s in zip(jobs, res)}
    with out.open("a") as f:
        for d, checks, judge_k in recs:
            f.write(json.dumps({"i": d["i"], "verdict": d["verdict"], "samples": d["samples"], "checks": checks,
                                "scores": {str(k): judged[(d["i"], k)] for k in judge_k}}, ensure_ascii=False) + "\n")
    print(f"scored {len(recs)} inputs, {len(jobs)} judge calls ({sum(s is None for s in res)} failures)", flush=True)


def cmd_pairs(a) -> None:
    data = Path(a.data)
    rows = _read(data / "train.jsonl")
    pairs, st = [], {"no_pair": 0, "chosen_from_student": 0, "chosen_from_teacher": 0, "rejected_unverified": 0,
                     "rejected_coverage": 0, "rejected_faithful": 0}
    for d in _read(data / a.scored):
        cand = [(int(k), s) for k, s in d["scores"].items() if s]
        if not cand:
            st["no_pair"] += 1
            continue
        c = d["checks"]
        good = [(k, s) for k, s in cand if rank(c[k])[0] and s["faithful"] >= 5 and s["coherent"] >= 4
                and s["consistent"] and s["language_ok"]]
        bad = [(k, s) for k, s in cand if not c[k]["verified"] or c[k]["errors"] or not c[k]["coverage"]
               or s["faithful"] <= a.max_rejected_faithful or s["coherent"] <= 2]
        if not bad:
            st["no_pair"] += 1  # nothing clearly bad to push away from
            continue
        k_bad = min(bad, key=lambda x: (rank(c[x[0]]), x[1]["faithful"] * 2 + x[1]["coherent"]))[0]
        teacher = rows[d["i"]]["output"].split("\n", 1)[1]
        if good:
            chosen = d["samples"][max(good, key=lambda x: (x[1]["faithful"] * 2 + x[1]["coherent"], rank(c[x[0]])))[0]]
            st["chosen_from_student"] += 1
        else:
            chosen = teacher
            st["chosen_from_teacher"] += 1
        if chosen == d["samples"][k_bad]:
            st["no_pair"] += 1
            continue
        cb = c[k_bad]
        st["rejected_unverified"] += (not cb["verified"]) or bool(cb["errors"])
        st["rejected_coverage"] += not cb["coverage"]
        st["rejected_faithful"] += d["scores"][str(k_bad)]["faithful"] <= a.max_rejected_faithful
        pairs.append({"i": d["i"], "verdict": d["verdict"], "chosen": chosen, "rejected": d["samples"][k_bad]})
    with (data / a.out).open("w") as f:
        for p in pairs:
            f.write(json.dumps(p, ensure_ascii=False) + "\n")
    print(f"{len(pairs)} pairs", st, flush=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(required=True)
    p = sub.add_parser("sample")
    p.add_argument("--run", required=True); p.add_argument("--data", required=True)
    p.add_argument("--n", type=int, default=15000); p.add_argument("--k", type=int, default=8)
    p.add_argument("--temp", type=float, default=0.8); p.add_argument("--batch", type=int, default=96)
    p.add_argument("--max-new", type=int, default=420); p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", default="onpolicy_c_samples.jsonl"); p.set_defaults(fn=cmd_sample)
    p = sub.add_parser("score")
    p.add_argument("--data", required=True); p.add_argument("--samples", default="onpolicy_c_samples.jsonl")
    p.add_argument("--out", default="onpolicy_c_scored.jsonl"); p.add_argument("--workers", type=int, default=192)
    p.set_defaults(fn=cmd_score)
    p = sub.add_parser("pairs")
    p.add_argument("--data", required=True); p.add_argument("--scored", default="onpolicy_c_scored.jsonl")
    p.add_argument("--out", default="dpo_c_pairs.jsonl"); p.add_argument("--max-rejected-faithful", type=int, default=3)
    p.set_defaults(fn=cmd_pairs)
    a = ap.parse_args()
    a.fn(a)


if __name__ == "__main__":
    main()
