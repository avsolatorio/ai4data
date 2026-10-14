"""Keep the rows of a rebuilt arm that are also in a reference arm, so a rebuilt arm holds the rows of the arms it is
compared with. A row matches when its input and its plain output (claims.plain_output) are the same; repeated rows are
matched as many times as the reference has them. Rows are kept in the rebuilt file's order (source order). Reference
rows the rebuilt arm does not have (its conversion failed this time) are counted and reported.

  python scripts/align_arm.py data/fmt_s2/C data/fmt_s/A data/fmt_s/C2
"""
import json
import sys
from collections import Counter
from pathlib import Path

from relevance_explainer.claims import plain_output

new, ref, out = map(Path, sys.argv[1:4])
out.mkdir(parents=True, exist_ok=True)
for split in ("train", "val", "test"):
    want = Counter()
    for line in open(ref / f"{split}.jsonl"):
        r = json.loads(line)
        want[(r["input"], plain_output(r["output"]))] += 1
    n_ref, kept, extra = sum(want.values()), 0, 0
    with open(out / f"{split}.jsonl", "w") as f:
        for line in open(new / f"{split}.jsonl"):
            r = json.loads(line)
            k = (r["input"], plain_output(r["output"]))
            if want[k] > 0:
                want[k] -= 1
                f.write(line)
                kept += 1
            else:
                extra += 1
    print(f"{split}: kept {kept} of {n_ref} reference rows ({n_ref - kept} missing), left out {extra} extra rows")
