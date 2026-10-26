"""Check generated training examples against the label set, and clean them.

Reads examples (JSON lines with text, labels, request_id, generator, and
optionally kind) and
the label file written by prepare_labels.py. Names listed in its
merged_from column are replaced by their label first. Prints one line per
check with the number of examples it affects:

    missing text          examples with an empty text
    background            examples of kind "background", which have no label on purpose
    no labels             other examples with an empty label list
    unknown label names   distinct label names that are not in the label set
    unknown label uses    times such a name was assigned
    left without label    examples whose labels are all unknown
    duplicate texts       repeats of a text already seen (case and spacing ignored)
    label name in text    examples whose text contains one of its labels verbatim
    background naming a label   background examples whose text contains any label name

It then prints the most frequent unknown names, the number of labels per
example, and the smallest and largest label counts in the cleaned set.
With --out, it writes the cleaned examples: unknown names removed,
examples with no text, no label, or a repeated text dropped, and
background examples that name a label dropped, because they would teach
the model that the name is no evidence of the topic. Topic examples
that name their label are kept and only reported.

Uses pandas.

Usage:
    python check_examples.py examples.jsonl labels.csv
    python check_examples.py examples.jsonl labels.csv --out clean.jsonl

Exit status: 0 always; the report is for a person to read.

What this does not check: whether the labels are right for the text,
or whether two texts say the same thing in different words. The
evaluation on real text in chapter 5 measures the first; a sample read
by the task owner catches the second.
"""

from __future__ import annotations

import argparse
import re
import sys
from collections import Counter

import pandas as pd


def label_set(labels_csv: str) -> tuple[set[str], dict[str, str]]:
    table = pd.read_csv(labels_csv, keep_default_na=False)
    merged = {
        old.strip(): row["label"]
        for _, row in table.iterrows()
        for old in row["merged_from"].split(";")
        if old.strip()
    }
    return set(table["label"]), merged


def check(
    examples: pd.DataFrame, known: set[str], merged: dict[str, str]
) -> tuple[dict, pd.DataFrame, Counter]:
    examples = examples.copy()
    if "kind" not in examples:
        examples["kind"] = "topic"
    background = examples["kind"].eq("background")
    examples["text"] = examples["text"].fillna("").astype(str).str.strip()
    examples["labels"] = examples["labels"].map(
        lambda ls: [merged.get(x, x) for x in (ls if isinstance(ls, list) else [])]
    )
    unknown = Counter(x for ls in examples["labels"] for x in ls if x not in known)
    examples["kept"] = examples["labels"].map(
        lambda ls: sorted({x for x in ls if x in known})
    )
    key = examples["text"].str.lower().map(lambda s: re.sub(r"\s+", " ", s))
    duplicate = key.duplicated() & (key != "")
    named = [
        any(label.lower() in text.lower() for label in labels)
        for text, labels in zip(examples["text"], examples["kept"])
    ]
    lowered = [label.lower() for label in known]
    background_named = background & examples["text"].str.lower().map(
        lambda text: any(label in text for label in lowered)
    )
    report = {
        "examples": len(examples),
        "missing text": int((examples["text"] == "").sum()),
        "background": int(background.sum()),
        "no labels": int(((examples["labels"].map(len) == 0) & ~background).sum()),
        "unknown label names": len(unknown),
        "unknown label uses": sum(unknown.values()),
        "left without label": int(
            ((examples["kept"].map(len) == 0) & (examples["labels"].map(len) > 0)).sum()
        ),
        "duplicate texts": int(duplicate.sum()),
        "label name in text": int(sum(named)),
        "background naming a label": int(background_named.sum()),
    }
    has_label = examples["kept"].map(len) > 0
    keep = (
        (examples["text"] != "")
        & (has_label | (background & ~background_named))
        & ~duplicate
    )
    clean = (
        examples.loc[keep].drop(columns=["labels"]).rename(columns={"kept": "labels"})
    )
    report["clean examples"] = len(clean)
    return report, clean, unknown


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("examples")
    parser.add_argument("labels")
    parser.add_argument("--out")
    args = parser.parse_args(argv)
    known, merged = label_set(args.labels)
    examples = pd.read_json(args.examples, lines=True)
    report, clean, unknown = check(examples, known, merged)
    for name, value in report.items():
        print(f"{name:<27}{value}")
    if unknown:
        print(
            "most frequent unknown names: "
            + ", ".join(f"{n} ({c})" for n, c in unknown.most_common(6))
        )
    per_example = clean["labels"].map(len).value_counts().sort_index()
    print(
        "labels per example: " + ", ".join(f"{k}: {v}" for k, v in per_example.items())
    )
    counts = Counter(x for ls in clean["labels"] for x in ls)
    missing = known - set(counts)
    rare = sorted(counts.items(), key=lambda item: item[1])
    print(f"labels with no example: {len(missing)}")
    print("fewest examples: " + ", ".join(f"{n} ({c})" for n, c in rare[:3]))
    print("most examples: " + ", ".join(f"{n} ({c})" for n, c in rare[-3:][::-1]))
    if args.out:
        clean.to_json(args.out, orient="records", lines=True, force_ascii=False)
        print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
