"""Does the mechanical check agree with the judge? Converts the deployed model's judged app answers to the claim
format and compares the judge's faithfulness for answers with and without unverified numbers or places, and with and
without a consistency flag.

  python scripts/claims_vs_judge.py runs/v2g-from-dapt-8ep-dpo/app_answers_realtest_judged.jsonl \
      data/eval_real/itemfirst_nosim_v4/test.jsonl
"""
import json
import sys

from relevance_explainer.claims import annotate, verify

judged = [json.loads(line) for line in open(sys.argv[1])]
inputs = [json.loads(line)["input"] for line in open(sys.argv[2])]
groups: dict[str, list[int]] = {}
for j in judged:
    if j.get("kind") != "final" or not j.get("judge") or j["verdict"] not in ("relevant", "partly_relevant", "irrelevant"):
        continue
    ck = verify(annotate(j["verdict"], j["narrative"], inputs[j["i"]]), inputs[j["i"]])
    f = j["judge"]["faithful"]
    groups.setdefault("unverified span" if ck.unverified else "no unverified span", []).append(f)
    groups.setdefault("consistency flag" if ck.flags else "no flag", []).append(f)
for g, fs in groups.items():
    print(f"{g:20s} n {len(fs):5d}  mean faithfulness {sum(fs) / len(fs):.2f}  rated 3 or lower {sum(x <= 3 for x in fs) / len(fs):.1%}")
