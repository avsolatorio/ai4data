"""Byte-level corpus shards: raw/<lang>.jsonl -> bytes/<lang>.bin (uint8 UTF-8 bytes, documents separated by EOS).

Byte values 0 to 255 are themselves; BOS = 256 and EOS = 257 exist only in the model's vocabulary. In the shard,
documents are separated by the byte 0x00 (which never occurs in the text), and the loader maps it to EOS.

  python -m relevance_explainer.tiny.bytecorpus --root data/corpus
"""
from __future__ import annotations

import argparse
import json
import multiprocessing as mp
from pathlib import Path

from .corpus import LANGS

SEP = 0  # document separator in the shard, mapped to EOS by the loader


def _one(args) -> tuple[str, int]:
    lang, root = args
    src, out = Path(root) / "raw" / f"{lang}.jsonl", Path(root) / "bytes" / f"{lang}.bin"
    n = 0
    with src.open() as f, out.open("wb") as g:
        for line in f:
            b = json.loads(line)["t"].encode("utf-8").replace(b"\x00", b"") + b"\x00"
            g.write(b)
            n += len(b)
    return lang, n


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    ap.add_argument("--procs", type=int, default=11)
    a = ap.parse_args()
    (Path(a.root) / "bytes").mkdir(parents=True, exist_ok=True)
    total = 0
    with mp.Pool(min(len(LANGS), a.procs)) as pool:
        for lang, n in pool.imap_unordered(_one, [(l, a.root) for l in LANGS]):
            total += n
            print(f"{lang}: {n / 1e6:.0f}M bytes", flush=True)
    print(f"total {total / 1e6:.0f}M bytes")


if __name__ == "__main__":
    main()
