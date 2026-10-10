"""Check that the app's answer confidence (features from decoding-time step statistics, CPU, INT8 export) matches the
teacher-forced features of tiny/selfcheck.py on the same realistic-test inputs, and time the scoring overhead.

  python scripts/check_answer_score.py runs/v2g-from-dapt-8ep-dpo 60
"""
import json
import sys
import time
from pathlib import Path

import torch

from relevance_explainer import viewer as V
from relevance_explainer.tiny.answer_score import features
from relevance_explainer.tiny.numguard import unsupported
from relevance_explainer.tiny.scriptguard import banned_for

RUN, N = Path(sys.argv[1]), int(sys.argv[2])
torch.set_num_threads(4)
name = "Realistic, no retrieval score"
model, tk, masks, _, _ = V.load(name)
head = V.score_head(name)
rows = [json.loads(l) for l in open("data/eval_real/itemfirst_nosim_v4/test.jsonl")]
F = torch.load(RUN / "selfcheck/test_features.pt", weights_only=False)
ref = {m["i"]: j for j, m in enumerate(F["meta"]) if m["k"] == 0}
samp = {d["i"]: d for d in map(json.loads, open(RUN / "selfcheck/test_samples.jsonl"))}
nl = tk.encode("\n").ids[-1]
diffs, same_text, n, t_dec, t_score = [], 0, 0, 0.0, 0.0
for i in sorted(ref)[:N]:
    text = rows[i]["input"]
    prompt = tk.encode(text + V.SEP).ids
    ban = banned_for(masks, text)
    toks, trace, probs = [], [], None
    t = time.time()
    for k, (tok, logits) in enumerate(V.stream(model, prompt, banned=ban, trace=trace)):
        if k == 0:
            probs = V.verdict_probs(tk, logits)
        if tok == V.EOS:
            break
        toks.append(tok)
    t_dec += time.time() - t
    out = tk.decode(toks).strip()
    t = time.time()
    vp = probs.get(out.split("\n")[0].strip(), max(probs.values()))
    f = features(trace, toks, nl, vp, bool(unsupported(out, text)))
    s_app = head(f)
    t_score += time.time() - t
    r = F["scalars"][ref[i]].tolist()
    s_ref = head(r)
    n += 1
    same_text += out == samp[i]["texts"][0]
    diffs.append(abs(s_app - s_ref))
    if n <= 3:
        print("app", [round(x, 3) for x in f], round(s_app, 3))
        print("ref", [round(x, 3) for x in r], round(s_ref, 3))
d = torch.tensor(diffs)
print(f"{n} inputs: same greedy text {same_text}; |score app - score ref| mean {d.mean():.4f}, max {d.max():.4f}; "
      f"diff > 0.02: {sum(1 for x, i in zip(diffs, sorted(ref)[:N]) if x > 0.02)}")
print(f"decoding {t_dec / n:.2f} s per answer; scoring {t_score / n * 1000:.2f} ms per answer")
