"""Add simulated retriever-scale variants to data/sim_variants: shifted up (a general-purpose embedding model whose
cosines sit higher), compressed, and both. Ranks are unchanged, only the scale moves.

  python scripts/sim_shifted.py
"""
import json
import re
from pathlib import Path

base = Path("data/sim_variants")
rows = [json.loads(l) for l in (base / "orig/test.jsonl").read_text().splitlines()]
VARIANTS = {"shift_up": lambda s: min(1.0, s + 0.3), "compress": lambda s: s * 0.5,
            "generic_like": lambda s: min(1.0, 0.45 + 0.5 * s)}  # typical general-model range 0.45 to 0.95
for name, f in VARIANTS.items():
    out = []
    for r in rows:
        m = re.search(r"Retrieval similarity: ([-0-9.]+)", r["input"])
        s = f(float(m.group(1)))
        out.append({**r, "input": r["input"].replace(m.group(0), f"Retrieval similarity: {s:.2f}")})
    d = base / name
    d.mkdir(exist_ok=True)
    text = "".join(json.dumps(x, ensure_ascii=False) + "\n" for x in out)
    (d / "test.jsonl").write_text(text)
    (d / "val.jsonl").write_text(text)
    (d / "train.jsonl").write_text(text.split("\n", 1)[0] + "\n")
    print(name, len(out))
