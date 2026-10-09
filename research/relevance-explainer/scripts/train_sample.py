"""Write a test set drawn from the training split: up to N rows per relation for the given origin, in the evaluation
schema, so the model's fit on its own training rows can be compared with its accuracy on the held-out suite.

  python scripts/train_sample.py data/v2_f_itemfirst_nosim data/trainfit_phen phen 150
"""
import json
import random
import sys
from collections import defaultdict
from pathlib import Path

src, dst, origin, n = Path(sys.argv[1]), Path(sys.argv[2]), sys.argv[3], int(sys.argv[4])
by = defaultdict(list)
for line in (src / "train.jsonl").open():
    r = json.loads(line)
    if r.get("origin") == origin and not r.get("repeat"):
        by[r.get("relation")].append(r)
rng = random.Random(0)
out = []
for rel, rs in sorted(by.items(), key=lambda x: str(x[0])):
    rng.shuffle(rs)
    for r in rs[:n]:
        out.append({**r, "majority": r["verdict"], "unanimous": True, "labels": {"stored": r["verdict"]}, "mode": rel, "style": None})
dst.mkdir(parents=True, exist_ok=True)
text = "".join(json.dumps(x, ensure_ascii=False) + "\n" for x in out)
for split in ("test", "val"):
    (dst / f"{split}.jsonl").write_text(text)
(dst / "train.jsonl").write_text(text.split("\n", 1)[0] + "\n")
print(len(out), {k: min(n, len(v)) for k, v in by.items()})
