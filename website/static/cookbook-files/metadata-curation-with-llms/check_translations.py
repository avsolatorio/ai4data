"""Check translated metadata fields before a native speaker reviews them.

Reads a translations file (CSV: record_id, field, language, source_text,
translated_text, status machine|reviewed, reviewer) and flags, per row,
the problems a reviewer should look at first:

    numbers changed   digits, years, or thresholds in the source are missing
                      or different in the translation (1000 -> 100)
    untranslated      the translation equals the source
    length            the translation is shorter than half or longer than
                      twice the source, which usually means dropped or
                      added content
    unreviewed        status is machine, so the text may not be published

Prints the flags and the counts, and exits 1 when any machine translation
has a numbers or length flag, so that a pipeline holds those back.
Standard library only.

Usage:
    python check_translations.py translations_example.csv

What this does not do: it checks form. A fluent translation that uses the
wrong statistical term ("chômage" for a concept that the organization
calls differently) passes every check here and is caught by the native
speaker with the vocabulary of the standardization chapter.
"""

from __future__ import annotations

import argparse
import csv
import re
import sys
from collections import Counter
from pathlib import Path

NUMBER = re.compile(r"\d+(?:[.,]\d+)?")


def numbers(text: str) -> Counter:
    return Counter(n.replace(",", ".") for n in NUMBER.findall(text))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("translations", type=Path)
    args = parser.parse_args(argv)
    with args.translations.open(newline="", encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh))

    counts = Counter()
    hold = 0
    print(f"{'record':<22} {'field':<17} {'lang':<4} {'status':<9} flags")
    for r in rows:
        flags = []
        src, dst = r["source_text"], r["translated_text"]
        if numbers(src) != numbers(dst):
            missing = numbers(src) - numbers(dst)
            added = numbers(dst) - numbers(src)
            flags.append(
                "numbers changed"
                + (f" (source {', '.join(missing)}" if missing else "")
                + (f"; translation {', '.join(added)}" if added else "")
                + (")" if missing or added else "")
            )
        if src.strip() == dst.strip():
            flags.append("untranslated")
        ratio = len(dst) / len(src) if src else 1.0
        if ratio < 0.5 or ratio > 2.0:
            flags.append(f"length ratio {ratio:.2f}")
        if r["status"] != "reviewed":
            flags.append("unreviewed")
        for f in flags:
            counts[f.split(" (")[0].split(" ratio")[0]] += 1
        if r["status"] != "reviewed" and any(
            f.startswith(("numbers", "length")) for f in flags
        ):
            hold += 1
        print(
            f"{r['record_id']:<22} {r['field']:<17} {r['language']:<4} {r['status']:<9} {'; '.join(flags) if flags else 'ok'}"
        )
    print(
        f"\n{len(rows)} translations: "
        + ", ".join(f"{k} {v}" for k, v in counts.most_common())
    )
    print(
        f"{hold} machine translation(s) held back for numbers or length; the rest go to the native speaker"
    )
    return 1 if hold else 0


if __name__ == "__main__":
    sys.exit(main())
