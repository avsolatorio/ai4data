"""Download the reports listed in sources.json and check their SHA-256.

    python fetch.py                 # download missing files, check every file
    python fetch.py --record        # write the SHA-256 and size of each file into sources.json
    python fetch.py --out DIR       # another download folder (default: local/rag-public/pdf)

The files are kept out of git. The default folder, local/rag-public/ at the
repository root, is excluded by the repository's local settings
(.git/info/exclude); on a clone without that setting, pass --out with a
folder under scripts/videos/out/, which .gitignore excludes.
"""

from __future__ import annotations

import argparse
import hashlib
import http.client
import json
import sys
import time
import urllib.request
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[3]
SOURCES = HERE / "sources.json"
DEFAULT_OUT = REPO / "local" / "rag-public" / "pdf"


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for block in iter(lambda: fh.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def download(url: str, target: Path) -> None:
    tmp = target.with_suffix(target.suffix + ".part")
    req = urllib.request.Request(url, headers={"User-Agent": "ai4data-rag-example/1.0"})
    for attempt in range(3):
        try:
            with urllib.request.urlopen(req, timeout=120) as resp, tmp.open("wb") as fh:
                while block := resp.read(1 << 20):
                    fh.write(block)
            tmp.replace(target)
            return
        except (OSError, http.client.HTTPException) as exc:  # network errors: retry twice
            print(f"  attempt {attempt + 1} failed: {exc}")
            time.sleep(3)
    raise SystemExit(f"could not download {url}")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT)
    ap.add_argument("--record", action="store_true", help="write sha256 and bytes into sources.json")
    args = ap.parse_args()

    manifest = json.loads(SOURCES.read_text())
    args.out.mkdir(parents=True, exist_ok=True)
    bad = 0
    for doc in manifest["documents"]:
        target = args.out / doc["file"]
        if not target.exists():
            print(f"downloading {doc['id']}: {doc['title']}")
            download(doc["pdf_url"], target)
        digest = sha256(target)
        if args.record:
            doc["sha256"], doc["bytes"] = digest, target.stat().st_size
            print(f"recorded  {doc['id']}  {digest[:16]}  {doc['bytes']:,} bytes")
        elif digest != doc.get("sha256"):
            bad += 1
            print(f"MISMATCH  {doc['id']}  expected {doc.get('sha256')}  got {digest}")
        else:
            print(f"ok        {doc['id']}  {digest[:16]}")
    if args.record:
        SOURCES.write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n")
    print(f"{len(manifest['documents'])} documents in {args.out}; {bad} mismatches")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
