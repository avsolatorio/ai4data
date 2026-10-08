"""Copy a data directory without rows whose narrative states a number absent from the input (train, val, test).

  python scripts/filter_numbers.py data/v2_e_itemfirst_nosim data/v2_f_itemfirst_nosim
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from relevance_explainer.tiny.numguard import unsupported  # noqa: E402

src, dst = Path(sys.argv[1]), Path(sys.argv[2])
dst.mkdir(parents=True, exist_ok=True)
for name in ("train", "val", "test"):
    n = kept = 0
    with (src / f"{name}.jsonl").open() as fi, (dst / f"{name}.jsonl").open("w") as fo:
        for line in fi:
            n += 1
            r = json.loads(line)
            if not unsupported(r["output"], r["input"]):
                fo.write(line)
                kept += 1
    print(f"{name}: kept {kept} of {n} ({n - kept} dropped)")
