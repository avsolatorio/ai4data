"""Build app/examples_realistic.json: 100 examples from the unseen realistic test set, with their catalog records so the
server can re-render inputs for edited searches.

  python -m relevance_explainer.webapp.realistic_examples
"""
from __future__ import annotations

import json
import random
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]


def main() -> None:
    rows = [json.loads(l) for l in (ROOT / "data/real_test/examples_labeled.jsonl").read_text().splitlines()]
    gen = {}
    p = ROOT / "runs/v2e-nosim-dpo/realtest_v3_generations_int8.jsonl"
    if p.exists():
        for r, g in zip(rows, [json.loads(l) for l in p.read_text().splitlines()]):
            gen[r["id"]] = g["generated"]
    rng = random.Random(5)
    rng.shuffle(rows)
    want = {"relevant": 25, "partly_relevant": 35, "irrelevant": 40}
    pick, by_v, by_l, by_s = [], Counter(), Counter(), Counter()
    for strict in (True, False):
        for r in rows:
            if len(pick) >= 100 or r in pick or by_v[r["verdict"]] >= want[r["verdict"]]:
                continue
            if strict and (by_l[r["language"]] >= 11 or by_s[r.get("style")] >= 35):
                continue
            pick.append(r)
            by_v[r["verdict"]] += 1
            by_l[r["language"]] += 1
            by_s[r.get("style")] += 1
    out = []
    for r in pick:
        out.append({"id": r["id"], "language": r["language"], "source": r["record"]["kind"], "query": r["query"],
                    "facets": r.get("facet_values") or {}, "record": r["record"], "similarity": r["similarity"],
                    "style": r.get("style"), "relation": r.get("relation"), "input": r["input_v2"], "input_v1": r["input_v1"],
                    "reference": f"{r['verdict']}\n{r['narrative']}",
                    "labels": {"stored": r["labels"]["stored"], "thinking": r["labels"]["thinking"] or r["verdict"], "gemma4": r["labels"]["gemma4"] or r["verdict"]},
                    "labels_missing": [k for k in ("thinking", "gemma4") if not r["labels"][k]],
                    "offline": {"text": gen.get(r["id"], ""), "judged": {}}, "unseen": True, "theme": r.get("topic")})
    (ROOT / "app/examples_realistic.json").write_text(json.dumps(out, ensure_ascii=False, indent=1))
    print(len(out), dict(by_v), dict(by_l), dict(by_s))


if __name__ == "__main__":
    main()
