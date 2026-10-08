"""Compare the data dictionaries of two survey rounds and draft a crosswalk.

Reads the dictionary of an earlier round and of a later round (the CSV
form used in this cookbook: name, label, type, values, question, ...) and
classifies every variable:

    same        same name, same value codes
    recoded     same name, different value codes: a break in the series
                until a correspondence is documented
    retyped     same name, different type
    renamed     different name, same label or question text
    dropped     in the earlier round only
    new         in the later round only

Writes the crosswalk as CSV (old_name, new_name, status, note) and prints
the counts. A duplicated name in either dictionary is reported and the
first occurrence is used. Standard library only.

Usage:
    python compare_rounds.py lfs_2024q4_dictionary.csv lfs_2025q2_dictionary.csv round_crosswalk.csv

What this does not do: it matches by name and by exact label or question
text. A variable renamed and relabelled at once is reported as dropped and
new, and a code list that kept its codes but changed their meaning is
reported as same; the methodologist's review of the crosswalk catches
both. A documented correspondence between code lists (an XKOS
correspondence table) is the full version of the "recoded" note.
"""

from __future__ import annotations

import argparse
import csv
import sys
from collections import Counter
from pathlib import Path


def load(path: Path) -> dict[str, dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh))
    counts = Counter(r["name"] for r in rows)
    for name, n in counts.items():
        if n > 1:
            print(
                f"warning: {path.name} lists {name!r} {n} times; first occurrence used"
            )
    out: dict[str, dict[str, str]] = {}
    for r in rows:
        out.setdefault(r["name"], r)
    return out


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("earlier", type=Path)
    parser.add_argument("later", type=Path)
    parser.add_argument("crosswalk", type=Path)
    args = parser.parse_args(argv)
    old = load(args.earlier)
    new = load(args.later)

    rows: list[dict[str, str]] = []
    matched_new: set[str] = set()
    for name, o in old.items():
        n = new.get(name)
        if n is not None:
            matched_new.add(name)
            if o.get("type") != n.get("type"):
                rows.append(
                    {
                        "old_name": name,
                        "new_name": name,
                        "status": "retyped",
                        "note": f"{o.get('type')} -> {n.get('type')}",
                    }
                )
            elif (o.get("values") or "") != (n.get("values") or ""):
                if not n.get("values"):
                    note = f"codes missing in {args.later.name}"
                elif not o.get("values"):
                    note = f"codes missing in {args.earlier.name}"
                else:
                    note = f"codes changed: {o.get('values')} -> {n.get('values')}"
                rows.append(
                    {
                        "old_name": name,
                        "new_name": name,
                        "status": "recoded",
                        "note": note,
                    }
                )
            else:
                rows.append(
                    {"old_name": name, "new_name": name, "status": "same", "note": ""}
                )
            continue
        candidates = [
            k
            for k, v in new.items()
            if k not in matched_new
            and (
                (o.get("label") and o.get("label") == v.get("label"))
                or (o.get("question") and o.get("question") == v.get("question"))
            )
        ]
        if candidates:
            k = candidates[0]
            matched_new.add(k)
            rows.append(
                {
                    "old_name": name,
                    "new_name": k,
                    "status": "renamed",
                    "note": f"matched on {'label' if o.get('label') == new[k].get('label') else 'question'}",
                }
            )
        else:
            rows.append(
                {"old_name": name, "new_name": "", "status": "dropped", "note": ""}
            )
    for name in new:
        if name not in matched_new:
            rows.append({"old_name": "", "new_name": name, "status": "new", "note": ""})

    with args.crosswalk.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(
            fh, fieldnames=["old_name", "new_name", "status", "note"]
        )
        writer.writeheader()
        writer.writerows(rows)
    counts = Counter(r["status"] for r in rows)
    print(
        f"{len(old)} variables in {args.earlier.name}, {len(new)} in {args.later.name}"
    )
    for status in ("same", "recoded", "retyped", "renamed", "dropped", "new"):
        if counts[status]:
            names = ", ".join(
                r["old_name"] or r["new_name"] for r in rows if r["status"] == status
            )
            print(f"{status:<8} {counts[status]:>3}  {names}")
    print(f"crosswalk written to {args.crosswalk}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
