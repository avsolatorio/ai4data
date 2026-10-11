"""Per-config train sizes of FineWeb-2 from the repository file listing (the dataset viewer's size endpoint fails for
some configs). Writes the same shape as the viewer's /size response, so multicorpus plan can read it.

  python scripts/fw2_sizes_from_hub.py <out.json>
"""
import json
import sys
from collections import defaultdict

from huggingface_hub import HfApi

size = defaultdict(int)
n = 0
for f in HfApi().list_repo_tree("HuggingFaceFW/fineweb-2", path_in_repo="data", repo_type="dataset", recursive=True):
    parts = f.path.split("/")  # data/<config>/<split>/<file>.parquet
    if len(parts) == 4 and parts[2] == "train" and f.path.endswith(".parquet"):
        size[parts[1]] += f.size
        n += 1
splits = [{"config": k, "split": "train", "num_bytes_parquet_files": v} for k, v in sorted(size.items())]
json.dump({"size": {"splits": splits}, "source": "huggingface_hub list_repo_tree"}, open(sys.argv[1], "w"))
print(f"{n} parquet files, {len(size)} configs, {sum(size.values()) / 1e12:.2f} TB")
for k in ("spa_Latn", "tur_Latn", "rus_Cyrl", "swh_Latn"):
    print(k, round(size.get(k, 0) / 1e6), "MB")
