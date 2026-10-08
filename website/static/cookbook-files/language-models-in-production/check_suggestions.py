"""Check model-suggested edit and imputation values against the edit rules.

Reads the edit rules (CSV: rule_id, field, rule range|consistency|lte,
parameters, message) and the model's suggestions (CSV: record_id, field,
current_value, suggested_value, reason, plus the record's other fields
the rules refer to) and reports, per suggestion, whether the suggested
value satisfies every rule on its field and the record, and whether the
reason names a field of the record (a grounded reason). Suggestions that
still fail a rule are listed for the editor; those that pass go to the
editor as proposals. Nothing is applied. Standard library only.

Usage:
    python check_suggestions.py edit_rules.csv suggested_values.csv

What this does not do: a value that passes the rules can still be wrong
(a plausible age for the wrong person). The rules bound the suggestion;
the editor decides; and the imputation method of the organization
(donor, model-based) stays the method of record, with the model's role
limited to proposing and explaining.
"""

from __future__ import annotations

import argparse
import csv
import re
import sys
from pathlib import Path


def value(
    record: dict[str, str], field: str, override: tuple[str, str] | None = None
) -> float | None:
    if override and override[0] == field:
        raw = override[1]
    else:
        raw = record.get(field, "")
    try:
        return float(raw)
    except ValueError:
        return None


def check(
    rule: dict[str, str], record: dict[str, str], field: str, suggested: str
) -> bool | None:
    override = (field, suggested)
    v = value(record, rule["field"], override)
    if rule["rule"] == "range":
        lo, hi = (float(x) for x in rule["parameters"].split(";"))
        return None if v is None else lo <= v <= hi
    if rule["rule"] == "lte":
        other = value(record, rule["parameters"], override)
        return None if v is None or other is None else v <= other
    if rule["rule"] == "consistency":
        m = re.match(r"(\w+)==(\w+)", rule["parameters"])
        if not m or v is None:
            return None
        status = record.get(m.group(1), "") if m.group(1) != field else suggested
        return status == m.group(2)
    return None


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("rules", type=Path)
    parser.add_argument("suggestions", type=Path)
    args = parser.parse_args(argv)
    with args.rules.open(newline="", encoding="utf-8") as fh:
        rules = list(csv.DictReader(fh))
    with args.suggestions.open(newline="", encoding="utf-8") as fh:
        suggestions = list(csv.DictReader(fh))
    fields = {
        k
        for s in suggestions
        for k in s
        if k not in ("record_id", "field", "current_value", "suggested_value", "reason")
    }

    passes = fails = 0
    print(f"{len(suggestions)} suggestions against {len(rules)} rules\n")
    for s in suggestions:
        record = {k: s[k] for k in fields}
        record[s["field"]] = s["suggested_value"]
        failed = []
        for rule in rules:
            if (
                rule["field"] != s["field"]
                and rule["parameters"].split(";")[0] != s["field"]
                and s["field"] not in rule["parameters"]
            ):
                continue
            ok = check(rule, record, s["field"], s["suggested_value"])
            if ok is False:
                failed.append(f"{rule['rule_id']} ({rule['message']})")
        grounded = any(
            f in s["reason"]
            for f in fields | {"roster", "respondent", "median", "status"}
        )
        verdict = (
            "to editor as proposal"
            if not failed
            else "still fails: " + "; ".join(failed)
        )
        passes += not failed
        fails += bool(failed)
        print(
            f"{s['record_id']:<9} {s['field']:<11} {s['current_value'] or '(empty)':>8} -> {s['suggested_value']:<6} {verdict}"
            + ("" if grounded else "  [reason does not cite the record]")
        )
    print(
        f"\n{passes} suggestions pass the rules and go to the editor; {fails} still fail and go back with the rule named; nothing is applied automatically"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
