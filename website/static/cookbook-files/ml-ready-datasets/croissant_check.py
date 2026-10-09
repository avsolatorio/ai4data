"""Check a Croissant record for the parts a loader and a catalog need.

Reads a Croissant JSON-LD file and checks: the dataset type and the
conformsTo declaration; the presence of name, description, url, license,
version, and citeAs; that every file in the distribution has a
contentUrl, an encodingFormat, and a sha256; that every record set has
fields, that every field has a dataType, and that every field source
names a file that exists in the distribution; and, when the files are
present beside the record, that each file's SHA-256 matches. Exit status
1 on any error. Standard library only.

Usage:
    python croissant_check.py lfs_occupation_ml.croissant.json

What this does not do: it checks structure and integrity. It does not
run the full Croissant validator (mlcroissant), does not load the data,
and does not judge whether the description or the licence are right.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

REQUIRED = ["name", "description", "url", "license", "version", "citeAs"]


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("record")
    parser.add_argument(
        "--files-dir",
        help="where the files named by contentUrl are; default: beside the record",
    )
    args = parser.parse_args(argv)

    path = Path(args.record)
    rec = json.loads(path.read_text(encoding="utf-8"))
    files_dir = Path(args.files_dir) if args.files_dir else path.parent
    errors: list[str] = []
    warnings: list[str] = []

    if rec.get("@type") != "sc:Dataset":
        errors.append("@type is not sc:Dataset")
    if not str(rec.get("conformsTo", "")).startswith("http://mlcommons.org/croissant/"):
        errors.append("conformsTo does not name a Croissant version")
    for k in REQUIRED:
        if not rec.get(k):
            errors.append(f"missing {k}")
    if rec.get("license") and not str(rec["license"]).startswith("http"):
        warnings.append(
            "license is not a URL; a URL is what tools and catalogs resolve"
        )

    file_ids: set[str] = set()
    for d in rec.get("distribution", []):
        ident = d.get("@id", "?")
        file_ids.add(ident)
        for k in ("contentUrl", "encodingFormat", "sha256"):
            if not d.get(k):
                errors.append(f"file {ident}: missing {k}")
        if (
            d.get("@type") == "cr:FileObject"
            and d.get("sha256")
            and d.get("contentUrl")
        ):
            local = files_dir / d["contentUrl"]
            if local.exists():
                if sha256(local) != d["sha256"]:
                    errors.append(f"file {ident}: sha256 does not match {local.name}")
            else:
                warnings.append(
                    f"file {ident}: {d['contentUrl']} not found beside the record; checksum not verified"
                )
    if not file_ids:
        errors.append("no distribution")

    sets = rec.get("recordSet", [])
    if not sets:
        errors.append("no recordSet")
    n_fields = 0
    for rs in sets:
        fields = rs.get("field", [])
        if not fields:
            errors.append(f"record set {rs.get('@id', '?')}: no fields")
        for fld in fields:
            n_fields += 1
            fid = fld.get("@id", "?")
            if not fld.get("dataType"):
                errors.append(f"field {fid}: missing dataType")
            src = fld.get("source")
            if src:
                ref = (src.get("fileObject") or src.get("fileSet") or {}).get("@id")
                if ref not in file_ids:
                    errors.append(f"field {fid}: source names unknown file {ref}")
            elif rs.get("data") is None:
                errors.append(
                    f"field {fid}: no source and the record set has no inline data"
                )

    print(
        f"{path.name}: {rec.get('name', '?')} version {rec.get('version', '?')}; {len(file_ids)} file(s), {len(sets)} record set(s), {n_fields} field(s)"
    )
    for w in warnings:
        print(f"  warning  {w}")
    for e in errors:
        print(f"  error    {e}")
    print(f"{len(errors)} error(s), {len(warnings)} warning(s)")
    return 1 if errors else 0


if __name__ == "__main__":
    sys.exit(main())
