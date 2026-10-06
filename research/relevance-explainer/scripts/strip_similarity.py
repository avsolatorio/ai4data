"""Copy a data directory with the "Retrieval similarity" line removed from every input (train, val, test).

  python scripts/strip_similarity.py data/v2_c_itemfirst data/v2_c_itemfirst_nosim
"""
import json
import re
import sys
from pathlib import Path

src, dst = Path(sys.argv[1]), Path(sys.argv[2])
dst.mkdir(parents=True, exist_ok=True)
pat = re.compile(r"\nRetrieval similarity: [-0-9.]+|^Retrieval similarity: [-0-9.]+\n")
for f in [src / f"{s}.jsonl" for s in ("train", "val", "test")]:  # the DPO side files are not needed
    n = k = 0
    with f.open() as fi, (dst / f.name).open("w") as fo:
        for line in fi:
            r = json.loads(line)
            new = pat.sub("", r["input"])
            k += new != r["input"]
            assert "Retrieval similarity" not in new
            r["input"] = new
            fo.write(json.dumps(r, ensure_ascii=False) + "\n")
            n += 1
    print(f"{f.name}: {n} rows, line removed in {k}")
