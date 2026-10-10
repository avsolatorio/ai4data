"""Run the app's generation stream (webapp.server._events) on realistic-test inputs and print the events after the
tokens: answer confidence, retry, attempts. Picks inputs whose greedy answer scored low and high in the selfcheck run.

  python scripts/check_retry_stream.py runs/v2g-from-dapt-8ep-dpo 2
"""
import json
import sys
from pathlib import Path

import torch

from relevance_explainer.webapp import server as S
from relevance_explainer import viewer as V

RUN, N = Path(sys.argv[1]), int(sys.argv[2])
name = "Realistic, no retrieval score"
head = V.score_head(name)
F = torch.load(RUN / "selfcheck/test_features.pt", weights_only=False)
greedy = [(m["i"], head(F["scalars"][j].tolist())) for j, m in enumerate(F["meta"]) if m["k"] == 0]
greedy.sort(key=lambda x: x[1])
rows = [json.loads(l) for l in open("data/eval_real/itemfirst_nosim_v4/test.jsonl")]
for i, sc in greedy[:N] + greedy[-1:]:
    print(f"\n=== input {i}: selfcheck score {sc:.3f}, teacher label {rows[i]['verdict']}")
    for line in S._events(rows[i]["input"], name, 0.0, True):
        ev = json.loads(line)
        if ev["type"] in ("token", "start"):
            continue
        if ev["type"] == "done":
            print("done:", ev["verdict"], "score", ev["score"], {k: ev["stats"][k] for k in ("retried", "first_score", "total_s")})
            print("  kept:", ev["narrative"][:160])
            for a in ev["attempts"]:
                print("  attempt", a.get("score"), "kept" if a["kept"] else "", a.get("note", ""), "|", a["narrative"][:120])
        else:
            print(ev)
