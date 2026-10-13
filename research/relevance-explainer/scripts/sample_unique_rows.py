"""A random sample of unique (input, output) pairs of a training file, written with all their repeats so the sample
keeps the file's weighting, and the mix of origins and languages it keeps compared with the full file.

  python scripts/sample_unique_rows.py data/v2_g_itemfirst_nosim/train.jsonl data/fmt_s/src/train.jsonl --n 250000
"""
import argparse
import json
import random
from collections import Counter
from pathlib import Path

from relevance_explainer.synth.tldr_grade import key

ap = argparse.ArgumentParser()
ap.add_argument("src"); ap.add_argument("out")
ap.add_argument("--n", type=int, default=250000)
ap.add_argument("--seed", type=int, default=0)
a = ap.parse_args()

rows = [json.loads(line) for line in open(a.src)]
keys = sorted({key(r) for r in rows})
keep = set(random.Random(a.seed).sample(keys, min(a.n, len(keys))))
Path(a.out).parent.mkdir(parents=True, exist_ok=True)
picked = [r for r in rows if key(r) in keep]
with open(a.out, "w") as f:
    for r in picked:
        f.write(json.dumps(r, ensure_ascii=False) + "\n")
share = lambda rs, k: {v: round(c / len(rs), 3) for v, c in Counter(r.get(k) for r in rs).most_common()}  # noqa: E731
print(f"{len(keep)} unique pairs, {len(picked)} rows (of {len(rows)})")
for k in ("origin", "language"):
    print(k, "sample", share(picked, k))
    print(k, "full  ", share(rows, k))
