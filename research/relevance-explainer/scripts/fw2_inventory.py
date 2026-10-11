"""Inventory of FineWeb-2 language configs by size, for choosing the multilingual base's languages.

  python scripts/fw2_inventory.py <fw2_size.json from datasets-server /size>

Prints how many language-script configs pass several size thresholds (train split, parquet bytes, which are compressed
text; roughly 2.5-3x more raw characters), the scripts they use, and writes the config list per threshold.
"""
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

d = json.loads(Path(sys.argv[1]).read_text())
by = defaultdict(int)
for c in d["size"]["splits"]:
    if c["split"] == "train" and not c["config"].endswith("_removed"):
        by[c["config"]] += c["num_bytes_parquet_files"]
print(f"{len(by)} configs with a train split")
items = sorted(by.items(), key=lambda x: -x[1])
for thr in (10e6, 50e6, 100e6, 300e6, 1e9):
    sel = [k for k, v in items if v >= thr]
    scripts = Counter(k.split("_")[1] for k in sel)
    print(f">= {thr / 1e6:6.0f} MB parquet: {len(sel):4d} configs, {len(scripts)} scripts, "
          f"{sum(by[k] for k in sel) / 1e9:.0f} GB; scripts: {dict(scripts.most_common(40))}")
    (Path(sys.argv[1]).parent / f"fw2_configs_ge{int(thr / 1e6)}MB.json").write_text(json.dumps({k: by[k] for k in sel}, indent=0))
print("smallest 15 above 50 MB:", [(k, round(by[k] / 1e6)) for k, v in items if v >= 50e6][-15:])
