"""Sample realistic test pairs as challenge searches, to measure how the verdict rules change ordinary labels.

  python scripts/policy_sample.py data/real_test/examples_labeled.jsonl data/policy_check_real 800
"""
import json
import random
import sys
from pathlib import Path

src, dst, n = sys.argv[1], Path(sys.argv[2]), int(sys.argv[3])
rows = [json.loads(l) for l in open(src)]
random.Random(5).shuffle(rows)
out = [{"id": f"rt-{i}", "phenomenon": f"realistic-{r.get('mode')}", "language": r["language"], "record": r["record"],
        "instruction": "", "query": r["query"], "writer_expected": r.get("verdict"), "note": "realistic test sample"}
       for i, r in enumerate(rows[:n])]
dst.mkdir(parents=True, exist_ok=True)
(dst / "searches.jsonl").write_text("".join(json.dumps(x, ensure_ascii=False) + "\n" for x in out))
print(len(rows), len(out))
