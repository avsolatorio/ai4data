"""Map model-assisted components to GSBPM phases and flag the risky ones.

Reads a component map (CSV: component, phase, sub_process, task,
model_role draft|flag|propose|code|explain|answer, review
always|sample|none, data_sensitivity public|internal|confidential,
model_location hosted|local, owner) and prints the components by GSBPM
phase, then the flags:

    unreviewed output    a role that produces text or codes users see
                         (answer, draft, code) with review "none"
    confidential hosted  confidential data sent to a hosted model
    deciding role        a role of "decide", which this guide does not
                         allow: models propose, flag, code, explain, and
                         draft; people decide

Exit status 1 when any flag is raised, so that the map can be checked in
a pipeline. Standard library only.

Usage:
    python map_components.py gsbpm_map.csv

What this does not do: the map records the organization's decisions; it
does not judge whether a task should use a model at all. The chapter's
typology and the evaluation suite of each component answer that, and the
GSBPM sub-process numbers follow version 5.1 and may differ in the
organization's own version.
"""

from __future__ import annotations

import argparse
import csv
import sys
from collections import defaultdict
from pathlib import Path

VISIBLE = {"answer", "draft", "code"}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("map", type=Path)
    args = parser.parse_args(argv)
    with args.map.open(newline="", encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh))
    by_phase: dict[str, list[dict[str, str]]] = defaultdict(list)
    for r in rows:
        by_phase[r["phase"]].append(r)
    print(f"{len(rows)} components across {len(by_phase)} phases\n")
    for phase, comps in by_phase.items():
        print(phase)
        for c in comps:
            print(
                f"  {c['sub_process']:<34} {c['component']:<28} {c['model_role']:<8} review {c['review']:<7} {c['data_sensitivity']:<13} {c['model_location']}"
            )
    flags = []
    for r in rows:
        if r["model_role"] in VISIBLE and r["review"] == "none":
            flags.append(
                f"{r['component']}: {r['model_role']} output reaches people with no review"
            )
        if r["data_sensitivity"] == "confidential" and r["model_location"] == "hosted":
            flags.append(f"{r['component']}: confidential data to a hosted model")
        if r["model_role"] == "decide":
            flags.append(
                f"{r['component']}: a model does not decide; change the role to propose or flag"
            )
    print()
    if flags:
        print("flags:")
        for f in flags:
            print(f"  {f}")
        return 1
    print("no flags: every visible output is reviewed and confidential data stay local")
    return 0


if __name__ == "__main__":
    sys.exit(main())
