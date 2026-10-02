"""Per-language subsets of an assembled dataset (train, val and test filtered to one language).

  python -m relevance_explainer.distill.split_language --data data/f1 --language French --out data/f1-fr
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--language", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    data, out = Path(a.data), Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    for split in ("train", "val", "test"):
        rows = [l for l in (data / f"{split}.jsonl").read_text().splitlines() if json.loads(l)["language"] == a.language]
        (out / f"{split}.jsonl").write_text("\n".join(rows) + ("\n" if rows else ""))
        print(f"{split}: {len(rows)}")


if __name__ == "__main__":
    main()
