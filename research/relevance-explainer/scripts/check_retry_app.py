"""Send each realistic app example through the app's own /api/generate route (in process, CPU), as the examples page
does, and report for low-confidence answers whether the retry ran and how many attempts came back.

  python scripts/check_retry_app.py [max_examples]
"""
import json
import sys

from starlette.testclient import TestClient

from relevance_explainer.webapp import server as S

name = "Realistic, no retrieval score"
c = TestClient(S.app)
n_low = n_retried = 0
for k, (eid, e) in enumerate(S._REALISTIC.items()):
    if len(sys.argv) > 1 and k >= int(sys.argv[1]):
        break
    body = {"input": e["input"], "model": name, "temperature": 0, "guard": True, "example_id": eid, "search": None}
    r = c.post("/api/generate", json=body)
    evs = [json.loads(l) for l in r.text.splitlines() if l.strip()]
    types = [ev["type"] for ev in evs if ev["type"] != "token"]
    done = next((ev for ev in evs if ev["type"] == "done"), None)
    if done is None:
        print(eid, "no done event", types, evs[-1])
        continue
    if done["score"] is not None and (done["score"] < 0.5 or done["stats"]["retried"]):
        n_low += 1
        n_retried += bool(done["stats"]["retried"])
        print(eid, "score", done["score"], "first", done["stats"]["first_score"], "retried", done["stats"]["retried"],
              "attempts", len(done["attempts"]), "not kept", sum(not a["kept"] for a in done["attempts"]), types)
print(f"low-confidence answers {n_low}, retried {n_retried}")
