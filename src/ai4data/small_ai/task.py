"""Task specification: what a classification task is, read from a task folder.

A task folder holds task.json and a label file. The specification states
the classification, the unit of text, whether a text takes one label or
several, how codes nest, how the generating model is asked for examples,
where the real reference data are, and which steps may send real text to
an outside service. Every other module reads the task from here, so that
the same steps serve topic tagging, occupation coding, and expenditure
coding.

The label file is a CSV with the columns

    id           the code or label used in the data, for example 2512
    name         its title, for example Software developers
    parent       the id of the broader label, if any
    description  optional: what the label covers
    includes     optional: examples of what it covers, separated by ";"
    excludes     optional: what it does not cover, separated by ";"
    aliases      optional: other names that map to this label, separated by ";"

Only the labels marked as leaves (by default, labels that are no other
label's parent) are predicted. For hierarchical codes, `levels` in
task.json gives the code lengths of the levels, for example [1, 2, 3, 4]
for ISCO-08, and the evaluation reports the accuracy at each level.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd


@dataclass
class Label:
    id: str
    name: str
    parent: str = ""
    description: str = ""
    includes: list[str] = field(default_factory=list)
    excludes: list[str] = field(default_factory=list)
    aliases: list[str] = field(default_factory=list)

    def prompt_line(self, detail: bool = True) -> str:
        """The label as the generating and labelling models see it."""
        head = self.name if self.id == self.name else f"{self.id} {self.name}"
        if not detail:
            return head
        parts = [head]
        if self.description:
            parts.append(self.description)
        if self.includes:
            parts.append("Includes: " + "; ".join(self.includes[:8]))
        if self.excludes:
            parts.append("Excludes: " + "; ".join(self.excludes[:5]))
        return " | ".join(parts)


@dataclass
class Task:
    folder: Path
    id: str
    title: str
    statement: str
    multi_label: bool
    labels: dict[str, Label]
    leaves: list[str]
    levels: list[int]
    generation: dict
    reference: dict
    model: dict
    privacy: dict
    documents: dict

    @property
    def leaf_labels(self) -> list[Label]:
        return [self.labels[i] for i in self.leaves]

    def alias_map(self) -> dict[str, str]:
        """Other names (and names in place of ids) mapped to leaf ids."""
        out = {}
        for label in self.leaf_labels:
            for alias in label.aliases:
                out[alias] = label.id
            out.setdefault(label.name, label.id)
        return out

    def ancestor(self, code: str, level: int) -> str:
        """The code at a level of the hierarchy, by code length (for example 2512 -> 25)."""
        return code[:level]

    def siblings(self, code: str) -> list[str]:
        parent = self.labels[code].parent
        return (
            [i for i in self.leaves if self.labels[i].parent == parent and i != code]
            if parent
            else []
        )

    def path(self, key: str) -> Path:
        return (self.folder / self.reference[key]).resolve()

    def allows_external(self, step: str) -> bool:
        """Whether real text may be sent to an outside service in this step."""
        return bool(self.privacy.get("real_text_to_external", {}).get(step, False))


def split_list(value) -> list[str]:
    if not isinstance(value, str) or not value.strip():
        return []
    return [part.strip() for part in value.split(";") if part.strip()]


def read_labels(path: Path) -> dict[str, Label]:
    table = pd.read_csv(path, dtype=str, keep_default_na=False)
    labels = {}
    for row in table.to_dict("records"):
        labels[row["id"]] = Label(
            id=row["id"],
            name=row.get("name") or row["id"],
            parent=row.get("parent", ""),
            description=row.get("description", ""),
            includes=split_list(row.get("includes", "")),
            excludes=split_list(row.get("excludes", "")),
            aliases=split_list(row.get("aliases", "")),
        )
    return labels


def load_task(folder: str | Path) -> Task:
    folder = Path(folder)
    if folder.is_file():
        folder = folder.parent
    spec = json.loads((folder / "task.json").read_text(encoding="utf-8"))
    labels = read_labels(folder / spec.get("labels", "labels.csv"))
    if spec.get("leaves") == "all":
        leaves = sorted(labels)
    else:
        parents = {label.parent for label in labels.values() if label.parent}
        leaves = sorted(i for i in labels if i not in parents)
    return Task(
        folder=folder.resolve(),
        id=spec["id"],
        title=spec["title"],
        statement=spec["statement"],
        multi_label=bool(spec.get("multi_label", False)),
        labels=labels,
        leaves=leaves,
        levels=list(spec.get("levels", [])),
        generation=spec.get("generation", {}),
        reference=spec.get("reference", {}),
        model=spec.get("model", {}),
        privacy=spec.get("privacy", {}),
        documents=spec.get("documents", {}),
    )
