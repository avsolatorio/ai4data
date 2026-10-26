"""Turn a published topic list into the label set of a classifier.

Reads a topic list in the form of wb_topics.json (a list "all_topics" in
page order and a mapping "topic_list" from each top-level topic to its
subtopics) and a merge file (CSV: from, to, reason) that names topics to
be treated as one label. Writes a label file (CSV: label, parents,
is_parent, merged_from) with one row per label, sorted by name, and prints
what changed between the list and the label set:

    entries on the list      every name on the list as published
    labels                   distinct names after the merges
    repeated names           names that appear more than once on the list
    merged names             names folded into another label
    several parents          labels listed under more than one topic

Uses pandas.

Usage:
    python prepare_labels.py wb_topics.json label_merges.csv labels.csv

Exit status: 0 when the label file is written, 1 when a merge names a
topic that is not on the list.

What this does not check: whether two labels with different names mean
the same thing, or whether a label is too broad to learn. Those are
decisions for the task owner, recorded in the merge file.
"""

from __future__ import annotations

import argparse
import json
import sys

import pandas as pd


def build_labels(topics: dict, merges: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    listed = topics["all_topics"]
    names = set(listed)
    unknown = sorted(set(merges["from"]) - names)
    if unknown:
        raise ValueError(f"merge names topics not on the list: {unknown}")
    target = dict(zip(merges["from"], merges["to"]))

    parents: dict[str, set[str]] = {}
    for parent, children in topics["topic_list"].items():
        parent = target.get(parent, parent)
        parents.setdefault(parent, set())
        for child in children:
            parents.setdefault(target.get(child, child), set()).add(parent)

    rows = []
    for label in sorted(parents):
        rows.append(
            {
                "label": label,
                "parents": "; ".join(sorted(parents[label] - {label})),
                "is_parent": "yes"
                if label in {target.get(p, p) for p in topics["topic_list"]}
                else "no",
                "merged_from": "; ".join(
                    sorted(k for k, v in target.items() if v == label)
                ),
            }
        )
    labels = pd.DataFrame(rows)
    counts = pd.Series(listed).value_counts()
    summary = {
        "entries on the list": len(listed),
        "labels": len(labels),
        "repeated names": int((counts > 1).sum()),
        "merged names": len(target),
        "several parents": int((labels["parents"].str.count(";") >= 1).sum()),
    }
    return labels, summary


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("topics")
    parser.add_argument("merges")
    parser.add_argument("out")
    args = parser.parse_args(argv)
    with open(args.topics, encoding="utf-8") as handle:
        topics = json.load(handle)
    try:
        labels, summary = build_labels(topics, pd.read_csv(args.merges))
    except ValueError as error:
        print(error, file=sys.stderr)
        return 1
    labels.to_csv(args.out, index=False)
    for key, value in summary.items():
        print(f"{key:<22}{value}")
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
