"""Metadata completeness report.

Reads a catalog export (CSV) and prints, for each field, the share of records
that have it filled, followed by the records that are missing a required
field. Standard library only.

The default required fields use the names of the World Bank indicator
metadata schema (worldbank/metadata-schemas, timeseries-schema.json), so a
catalog exported from NADA or the Metadata Editor can be checked as is.

Usage:
    python completeness_report.py example_catalog.csv
    python completeness_report.py catalog.csv --required id,title,description,unit
"""

import argparse
import csv
import sys

DEFAULT_REQUIRED = [
    "idno",
    "name",
    "definition_long",
    "measurement_unit",
    "periodicity",
    "time_period_start",
    "time_period_end",
    "geographic_units",
    "sources",
    "date_last_update",
]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("catalog", help="CSV export of the catalog")
    parser.add_argument(
        "--required",
        default=",".join(DEFAULT_REQUIRED),
        help="comma-separated list of fields that must be filled",
    )
    args = parser.parse_args()
    required = [f.strip() for f in args.required.split(",") if f.strip()]

    with open(args.catalog, newline="", encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh))
    if not rows:
        sys.exit("no records found")

    fields = list(rows[0].keys())
    print(f"{len(rows)} records, {len(fields)} fields\n")
    print(f"{'field':<22} {'filled':>7} {'share':>7}")
    for field in fields:
        filled = sum(1 for r in rows if (r.get(field) or "").strip())
        flag = "" if field not in required else " *"
        print(f"{field:<22} {filled:>7} {filled / len(rows):>7.0%}{flag}")
    print("\n* required field")

    incomplete = []
    for r in rows:
        missing = [f for f in required if not (r.get(f) or "").strip()]
        if missing:
            incomplete.append((r.get("idno", "?"), missing))

    print(f"\n{len(incomplete)} of {len(rows)} records missing a required field")
    for rid, missing in incomplete:
        print(f"  {rid}: {', '.join(missing)}")


if __name__ == "__main__":
    main()
