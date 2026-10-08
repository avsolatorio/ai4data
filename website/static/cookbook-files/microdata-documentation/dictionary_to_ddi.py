"""Convert a data dictionary CSV into the World Bank microdata schema.

Produces the `data_files` and `variables` sections of a microdata record
(worldbank/metadata-schemas, microdata-schema.json, which follows DDI
Codebook), and merges them into an existing study record when one is given.
The result can be imported into NADA or the Metadata Editor, or validated
with the metadata checker from the dissemination cookbook.

Mapping from the CSV columns to DDI fields:

    name      -> name            values    -> var_catgry [{value, label}]
    label     -> labl            missing   -> var_catgry entries flagged in var_notes
    type      -> var_format.type question  -> var_qstn_qstnlit
    universe  -> var_universe    concept   -> var_concept [{title}]

Standard library only.

What this does not do: it does not compute summary statistics (var_sumstat)
or category frequencies, which need the data file, and it does not validate
the result against the schema; run check_metadata.py from the dissemination
cookbook for that.

Usage:
    python dictionary_to_ddi.py lfs_2025q2_dictionary.csv --study example_microdata.json -o study_with_variables.json
    python dictionary_to_ddi.py lfs_2025q2_dictionary.csv            # prints the sections only
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

FORMAT = {"numeric": "numeric", "categorical": "numeric", "string": "character"}


def parse_values(text: str) -> list[dict[str, str]]:
    out = []
    for item in (text or "").split(";"):
        if "=" in item:
            value, label = item.split("=", 1)
            out.append({"value": value.strip(), "label": label.strip()})
    return out


def convert(rows: list[dict[str, str]]) -> tuple[list[dict], list[dict]]:
    files: dict[str, dict] = {}
    variables: list[dict] = []
    for i, r in enumerate(rows, 1):
        file_id = r["file_id"].strip()
        files.setdefault(file_id, {"file_id": file_id, "file_name": f"{file_id}.csv", "var_count": 0})
        files[file_id]["var_count"] += 1
        var: dict = {
            "vid": f"V{i}",
            "file_id": file_id,
            "name": r["name"].strip(),
            "labl": (r.get("label") or "").strip(),
            "var_format": {"type": FORMAT.get((r.get("type") or "").strip().lower(), "character")},
        }
        if (r.get("universe") or "").strip():
            var["var_universe"] = r["universe"].strip()
        if (r.get("question") or "").strip():
            var["var_qstn_qstnlit"] = r["question"].strip()
        categories = parse_values(r.get("values") or "")
        missing = [m.strip() for m in (r.get("missing") or "").split(";") if m.strip()]
        for code in missing:
            categories.append({"value": code, "label": "Missing"})
        if categories:
            var["var_catgry"] = categories
        if missing:
            var["var_notes"] = "Missing-value codes: " + ", ".join(missing)
        if (r.get("concept") or "").strip():
            var["var_concept"] = [{"title": r["concept"].strip()}]
        variables.append(var)
    return list(files.values()), variables


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("dictionary", type=Path)
    parser.add_argument("--study", type=Path, help="study record (JSON) to merge the sections into")
    parser.add_argument("-o", "--output", type=Path, help="where to write the result (default: stdout)")
    args = parser.parse_args(argv)

    with args.dictionary.open(newline="", encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh))
    files, variables = convert(rows)

    if args.study:
        with args.study.open(encoding="utf-8") as fh:
            record = json.load(fh)
        record["data_files"] = files
        record["variables"] = variables
    else:
        record = {"data_files": files, "variables": variables}

    text = json.dumps(record, indent=2, ensure_ascii=False)
    if args.output:
        args.output.write_text(text + "\n", encoding="utf-8")
        print(f"wrote {args.output}: {len(files)} file(s), {len(variables)} variable(s)")
    else:
        print(text)
    return 0


if __name__ == "__main__":
    sys.exit(main())
