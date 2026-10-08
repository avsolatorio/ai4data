"""Score mention extraction against a labelled sample.

Reads the extractor's mentions (JSON lines) and a labelled sample (CSV:
document_id, sentence, mention_text, with mention_text empty when the
sentence contains no dataset mention) and reports precision, recall, and
F1 at the level of (document, mention text), with the false positives and
false negatives listed. Mentions are compared after normalization (lower
case, punctuation removed). Standard library only.

Usage:
    python score_mentions.py mentions_example.jsonl labelled_sample.csv

What this does not do: fourteen labelled sentences give a demonstration.
A measure needs a sample of a few hundred sentences drawn across document
types, with two annotators and an agreement check, and the comparison
should also score the usage flag and the context, which this script leaves
out.
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import sys
from pathlib import Path

WORD = re.compile(r"[a-z0-9]+")


def norm(text: str) -> str:
    return " ".join(WORD.findall(text.lower()))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("mentions", type=Path)
    parser.add_argument("labelled", type=Path)
    args = parser.parse_args(argv)

    with args.mentions.open(encoding="utf-8") as fh:
        found = {
            (m["document_id"], norm(m["text"]))
            for m in (json.loads(line) for line in fh if line.strip())
        }
    with args.labelled.open(newline="", encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh))
    expected = {
        (r["document_id"], norm(r["mention_text"]))
        for r in rows
        if (r.get("mention_text") or "").strip()
    }

    tp = found & expected
    fp = found - expected
    fn = expected - found
    precision = len(tp) / len(found) if found else 0.0
    recall = len(tp) / len(expected) if expected else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0

    print(
        f"labelled mentions {len(expected)}, extracted {len(found)}, correct {len(tp)}"
    )
    print(f"precision {precision:.2f}  recall {recall:.2f}  F1 {f1:.2f}")
    if fp:
        print("false positives (extracted, not labelled as a mention):")
        for d, t in sorted(fp):
            print(f"  {d}: {t!r}")
    if fn:
        print("false negatives (labelled, not extracted):")
        for d, t in sorted(fn):
            print(f"  {d}: {t!r}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
