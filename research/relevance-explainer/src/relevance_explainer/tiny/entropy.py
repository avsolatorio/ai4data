"""Information budget of the narratives: teacher self-surprisal against student cross-entropy, in bits per character.

For each sampled validation example:
- teacher: log-probability the teacher assigns to its own narrative (in the exact JSON form it produced) given the
  narration prompt, via the vLLM completions endpoint with echoed prompt log-probabilities;
- student: teacher-forced log-probability of the same narrative under a trained student, given the student input and
  the verdict line;
- copyable share: fraction of narrative words (4+ letters) that occur in the input text.
Bits per character makes the two comparable despite different tokenizers.

  TEACHER_URLS=http://localhost:8000/v1 python -m relevance_explainer.tiny.entropy --run runs/f2-pre2 --data data/f2 --n 600
"""
from __future__ import annotations

import argparse
import glob
import json
import math
import random
import re
import unicodedata
import urllib.request
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import torch
from tokenizers import Tokenizer

from ..distill.teacher_checks import MODEL, strip_similarity, urls
from ..synth.generate import NARRATE_SYSTEM
from .model import Config, TinyLM

SEP = "\n\n### Judgment\n"
EOS = 1


def teacher_bits(hf_tok, base: str, inp: str, verdict: str, narrative: str) -> float | None:
    """Sum of -log2 p over the narrative's tokens under the teacher."""
    msgs = [{"role": "system", "content": NARRATE_SYSTEM}, {"role": "user", "content": strip_similarity(inp)}]
    prefix = hf_tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True, enable_thinking=False)
    head = prefix + json.dumps({"verdict": verdict}, ensure_ascii=False)[:-1] + ', "narrative": "'
    full = head + json.dumps(narrative, ensure_ascii=False)[1:-1] + '"}'
    n_head = len(hf_tok(head)["input_ids"])
    n_full = len(hf_tok(full)["input_ids"])
    body = {"model": MODEL, "prompt": full, "max_tokens": 1, "echo": True, "logprobs": 1, "prompt_logprobs": 0, "temperature": 0}
    try:
        req = urllib.request.Request(base + "/completions", json.dumps(body).encode(), {"Content-Type": "application/json"})
        pl = json.load(urllib.request.urlopen(req, timeout=300))["choices"][0]["prompt_logprobs"]
    except Exception:
        return None
    # the last token is '"}' (closing); narrative tokens are positions n_head .. n_full-2
    lps = []
    for entry in pl[n_head : n_full - 1]:
        if entry:
            lps.append(next(iter(entry.values()))["logprob"])
    return -sum(lps) / math.log(2) if lps else None


@torch.no_grad()
def student_bits(model: TinyLM, tk: Tokenizer, inp: str, verdict: str, narrative: str) -> float:
    p = tk.encode(inp + SEP + verdict + "\n").ids
    t = tk.encode(narrative).ids
    ids = torch.tensor([p + t]).cuda()
    with torch.autocast("cuda", dtype=torch.bfloat16):
        logits = model(ids)
    lp = torch.log_softmax(logits[0, len(p) - 1 : len(p) + len(t) - 1].float(), dim=-1)
    tok_lp = lp.gather(1, torch.tensor(t, device=lp.device)[:, None]).squeeze(1)
    return float(-tok_lp.sum() / math.log(2))


def copy_share(narrative: str, inp: str) -> float:
    norm = lambda s: unicodedata.normalize("NFKD", s).lower()
    words = [w for w in re.findall(r"\w{4,}", norm(narrative))]
    src = norm(inp)
    return sum(w in src for w in words) / max(len(words), 1)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--data", required=True)
    ap.add_argument("--split", default="val")
    ap.add_argument("--n", type=int, default=600)
    ap.add_argument("--workers", type=int, default=32)
    a = ap.parse_args()

    rows = [json.loads(l) for l in (Path(a.data) / f"{a.split}.jsonl").read_text().splitlines()]
    rows = random.Random(0).sample(rows, min(a.n, len(rows)))
    for r in rows:
        r["narrative"] = r["output"].split("\n", 1)[1]

    from transformers import AutoTokenizer
    snap = glob.glob(str(Path.home() / ".cache/huggingface/hub/models--Qwen--Qwen3.8-27B-FP8/snapshots/*"))[0]
    hf_tok = AutoTokenizer.from_pretrained(snap)
    servers = urls()
    with ThreadPoolExecutor(a.workers) as ex:
        tb = list(ex.map(lambda k: teacher_bits(hf_tok, servers[k % len(servers)], rows[k]["input"], rows[k]["verdict"], rows[k]["narrative"]), range(len(rows))))

    ck = torch.load(Path(a.run) / "model_fp32.pt", map_location="cpu")
    model = TinyLM(Config(**ck["config"]))
    model.load_state_dict(ck["state"])
    model = model.cuda().eval()
    tk = Tokenizer.from_file(str(Path(a.run) / "tokenizer.json"))
    sb = [student_bits(model, tk, r["input"], r["verdict"], r["narrative"]) for r in rows]

    groups: dict[str, list] = defaultdict(list)
    for r, t, s in zip(rows, tb, sb):
        if t is None:
            continue
        rec = (t, s, len(r["narrative"]), copy_share(r["narrative"], r["input"]))
        groups["all"].append(rec)
        if r["language"] != "English":
            groups["other languages"].append(rec)
        groups[r["language"]].append(rec)  # "English" is the English group; copy share uses \w{4,} so it is near zero for Chinese
    print(f"{'group':16s} {'n':>4s} {'teacher b/char':>14s} {'student b/char':>14s} {'teacher b/narr':>14s} {'student b/narr':>14s} {'copy share':>10s}")
    out = {}
    for g, recs in groups.items():
        if len(recs) < 15:
            continue
        T, S, C, P = (sum(x[i] for x in recs) for i in range(4))
        out[g] = {"n": len(recs), "teacher_bits_per_char": T / C, "student_bits_per_char": S / C,
                  "teacher_bits_per_narrative": T / len(recs), "student_bits_per_narrative": S / len(recs), "copy_share": P / len(recs)}
        o = out[g]
        print(f"{g:16s} {o['n']:4d} {o['teacher_bits_per_char']:14.3f} {o['student_bits_per_char']:14.3f} {o['teacher_bits_per_narrative']:14.1f} {o['student_bits_per_narrative']:14.1f} {o['copy_share']:10.2f}")
    (Path(a.run) / "entropy.json").write_text(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
