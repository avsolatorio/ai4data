"""Checks on generated examples, and the cleaned set.

The report counts, for a set of examples:

    examples                 rows read
    missing text             rows with an empty text
    background               rows of kind "background", which have no label on purpose
    no labels                other rows with an empty label list
    unknown label names      distinct labels that are not in the task's label file
    unknown label uses       times such a label was assigned
    left without label       rows whose labels are all unknown
    more than one label      rows with several labels in a single-label task
    duplicate texts          repeats of a text already seen (case and spacing ignored)
    label name in text       rows whose text contains the name of one of its labels
    background naming a label  background rows that contain a label name
    labeller disagreed       rows the labeller assigned differently (when verified)
    labeller failed          rows whose labeller reply could not be read

The cleaned set drops rows with no text, no known label (except
background rows), several labels in a single-label task, a repeated text,
a background row that names a label, and, with require_verified, a row
the labeller disagreed with. Rows that name their label are kept and
only counted.
"""

from __future__ import annotations

import re
from collections import Counter

import pandas as pd

from .task import Task


def check(
    examples: pd.DataFrame, task: Task, require_verified: bool = True
) -> tuple[dict, pd.DataFrame, Counter]:
    examples = examples.copy()
    if "kind" not in examples:
        examples["kind"] = "topic"
    background = examples["kind"].eq("background")
    known = set(task.leaves)
    aliases = task.alias_map()
    examples["text"] = examples["text"].fillna("").astype(str).str.strip()
    examples["labels"] = examples["labels"].map(
        lambda ls: [
            aliases.get(str(x), str(x)) for x in (ls if isinstance(ls, list) else [])
        ]
    )
    unknown = Counter(x for ls in examples["labels"] for x in ls if x not in known)
    examples["kept"] = examples["labels"].map(
        lambda ls: sorted({x for x in ls if x in known})
    )
    key = examples["text"].str.lower().map(lambda s: re.sub(r"\s+", " ", s))
    duplicate = key.duplicated() & (key != "")
    names = {x: task.labels[x].name.lower() for x in known}
    named = [
        any(names[x] in text.lower() for x in labels)
        for text, labels in zip(examples["text"], examples["kept"])
    ]
    lowered = list(names.values())
    background_named = background & examples["text"].str.lower().map(
        lambda t: any(n in t for n in lowered)
    )
    several = (
        (~examples["kept"].map(len).le(1))
        if not task.multi_label
        else pd.Series(False, index=examples.index)
    )
    if "verified" in examples:
        verified = examples["verified"]
        disagreed = verified.eq(False)
        failed = verified.isna()
    else:
        disagreed = failed = pd.Series(False, index=examples.index)
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
        "more than one label": int(several.sum()),
        "duplicate texts": int(duplicate.sum()),
        "label name in text": int(sum(named)),
        "background naming a label": int(background_named.sum()),
        "labeller disagreed": int(disagreed.sum()),
        "labeller failed": int(failed.sum()) if "verified" in examples else 0,
    }
    has_label = examples["kept"].map(len) > 0
    keep = (
        (examples["text"] != "")
        & (has_label | (background & ~background_named))
        & ~duplicate
        & ~several
    )
    if require_verified:
        keep &= ~disagreed
    clean = (
        examples.loc[keep].drop(columns=["labels"]).rename(columns={"kept": "labels"})
    )
    report["clean examples"] = len(clean)
    return report, clean, unknown


def agreement(examples: pd.DataFrame) -> pd.DataFrame:
    """Per label: verified rows, rows the labeller disagreed with, and the agreement rate."""
    if "verified" not in examples:
        return pd.DataFrame(columns=["label", "verified", "disagreed", "agreement"])
    rows = examples[examples["verified"].notna() & ~examples["kind"].eq("background")]
    written = rows["written_labels"] if "written_labels" in rows else rows["labels"]
    out = Counter(), Counter()
    for labels, ok in zip(written, rows["verified"]):
        for x in labels:
            out[0 if ok else 1][x] += 1
    table = (
        pd.DataFrame({"verified": pd.Series(out[0]), "disagreed": pd.Series(out[1])})
        .fillna(0)
        .astype(int)
    )
    table["agreement"] = (
        table["verified"] / (table["verified"] + table["disagreed"]).clip(lower=1)
    ).round(3)
    return table.rename_axis("label").reset_index()


def format_report(report: dict) -> str:
    return "\n".join(f"{name:<27}{value}" for name, value in report.items())
