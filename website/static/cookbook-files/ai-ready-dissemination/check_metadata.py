"""Check indicator metadata in three layers.

1. Structure: each record is validated against the World Bank indicator
   metadata schema (worldbank/metadata-schemas, timeseries-schema.json).
   The schema itself requires only `idno` and `name`, so passing this layer
   means the record is well-formed, and nothing more.
2. Completeness: each record is checked against a profile, a JSON file in
   which the office lists the fields it requires at each maturity level.
   The schema does not decide what is complete; the office does.
3. Validity: dates parse, periods are ordered, URLs look like URLs, the
   periodicity comes from the profile's vocabulary, and the definition is
   more than a repeat of the name.

What this does not check: whether the definition is correct, whether the
name and the definition describe the same thing, whether the unit matches
the values, or whether the text is specific enough to be useful. Those are
semantic checks; see the program's Generative AI for Metadata Quality work.

Input: a CSV export with one row per series and columns named as in the
schema (flat), or a JSON file with a list of schema records (nested).
Flat columns that hold lists use ";" between values, and
time_period_start / time_period_end map to time_periods.

Usage:
    python check_metadata.py example_catalog.csv completeness_profile_indicator.json
    python check_metadata.py records.json profile.json --schema timeseries-schema.json
    python check_metadata.py catalog.csv profile.json --level ai-ready

Needs the `jsonschema` package for layer 1 (pip install jsonschema). Without
it, layer 1 is skipped and the report says so.
"""

import argparse
import csv
import json
import re
import sys
import urllib.request
from collections import Counter
from pathlib import Path

SCHEMA_URL = (
    "https://raw.githubusercontent.com/worldbank/metadata-schemas/main/schemas/timeseries-schema.json"
)
SCHEMA_CACHE = Path(__file__).with_name("timeseries-schema.json")

# Flat column -> (schema field, property of the list item). Values are split
# on ";" and each becomes one item.
LIST_FIELDS = {
    "sources": "name",
    "geographic_units": "name",
    "ref_country": "name",
    "contacts": "email",
    "license": "uri",
    "keywords": "name",
    "topics": "name",
    "aliases": "alias",
    "languages": "code",
    "concepts": "name",
    "related_indicators": "code",
    "links": "uri",
    "definition_references": "uri",
    "methodology_references": "uri",
    "authoring_entity": "name",
}

DATE = re.compile(r"^\d{4}-\d{2}-\d{2}$")
PERIOD = re.compile(r"^\d{4}(-\d{2}(-\d{2})?|-Q[1-4]|-W\d{2}|-S[12])?$")
URL = re.compile(r"^https?://\S+$")
IDNO = re.compile(r"^[A-Za-z0-9_.\-]+$")


def load_records(path):
    """Return a list of nested schema records."""
    if path.endswith(".json"):
        data = json.load(open(path, encoding="utf-8"))
        return data if isinstance(data, list) else [data]
    records = []
    with open(path, newline="", encoding="utf-8") as fh:
        for row in csv.DictReader(fh):
            sd = {}
            tp = {}
            for col, raw in row.items():
                val = (raw or "").strip()
                if not val:
                    continue
                if col == "time_period_start":
                    tp["start"] = val
                elif col == "time_period_end":
                    tp["end"] = val
                elif col in LIST_FIELDS:
                    prop = LIST_FIELDS[col]
                    sd[col] = [{prop: v.strip()} for v in val.split(";") if v.strip()]
                else:
                    sd[col] = val
            if tp:
                sd["time_periods"] = [tp]
            records.append({"series_description": sd})
    return records


def load_schema(path):
    if path:
        return json.load(open(path, encoding="utf-8"))
    if SCHEMA_CACHE.exists():
        return json.load(open(SCHEMA_CACHE, encoding="utf-8"))
    try:
        with urllib.request.urlopen(SCHEMA_URL, timeout=20) as resp:
            text = resp.read().decode("utf-8")
        SCHEMA_CACHE.write_text(text, encoding="utf-8")
        return json.loads(text)
    except Exception as exc:  # noqa: BLE001
        print(f"schema not available ({exc}); pass --schema to use a local copy")
        return None


def layer_structure(records, schema):
    try:
        import jsonschema
    except ImportError:
        return None, ["jsonschema package not installed; layer 1 skipped"]
    if schema is None:
        return None, ["schema not loaded; layer 1 skipped"]
    import warnings

    with warnings.catch_warnings():
        # The schema names a metaschema that jsonschema may not know; the
        # latest draft is used in that case.
        warnings.simplefilter("ignore", DeprecationWarning)
        validator = jsonschema.validators.validator_for(schema)(schema)
    problems = []
    for rec in records:
        rid = rec.get("series_description", {}).get("idno", "?")
        for err in validator.iter_errors(rec):
            where = "/".join(str(p) for p in err.absolute_path)
            problems.append(f"{rid}: {where}: {err.message[:90]}")
    return len(records) - len({p.split(':')[0] for p in problems}), problems


def present(sd, field):
    val = sd.get(field)
    if val is None:
        return False
    if isinstance(val, str):
        return bool(val.strip())
    if isinstance(val, list):
        return len(val) > 0
    return True


def layer_completeness(records, profile, level):
    levels = profile["levels"]
    order = list(levels)
    wanted = []
    for lvl in order[: order.index(level) + 1]:
        wanted += levels[lvl]
    missing_by_record = {}
    missing_count = Counter()
    for rec in records:
        sd = rec.get("series_description", {})
        missing = [f for f in wanted if not present(sd, f)]
        if missing:
            missing_by_record[sd.get("idno", "?")] = missing
            missing_count.update(missing)
    return wanted, missing_by_record, missing_count


def layer_validity(records, profile):
    issues = []
    seen = Counter()
    vocab = profile.get("vocabularies", {})
    for rec in records:
        sd = rec.get("series_description", {})
        rid = sd.get("idno", "?")
        seen[rid] += 1
        if not IDNO.match(rid):
            issues.append(f"{rid}: idno has characters outside A-Z, 0-9, _ . -")
        d = sd.get("date_last_update")
        if d and not DATE.match(d):
            issues.append(f"{rid}: date_last_update {d!r} is not YYYY-MM-DD")
        for tp in sd.get("time_periods", []):
            s, e = tp.get("start"), tp.get("end")
            for v in (s, e):
                if v and not PERIOD.match(v):
                    issues.append(f"{rid}: period {v!r} is not an ISO 8601 or SDMX period")
            if s and e and PERIOD.match(s) and PERIOD.match(e) and e < s:
                issues.append(f"{rid}: time period ends ({e}) before it starts ({s})")
        p = sd.get("periodicity")
        if p and vocab.get("periodicity") and p not in vocab["periodicity"]:
            issues.append(f"{rid}: periodicity {p!r} is not in the profile vocabulary")
        for field in ("definition_references", "methodology_references", "links", "license"):
            for item in sd.get(field, []):
                uri = item.get("uri")
                if uri and not URL.match(uri):
                    issues.append(f"{rid}: {field} uri {uri!r} is not a URL")
        name = (sd.get("name") or "").strip().lower()
        definition = (sd.get("definition_long") or "").strip()
        if definition:
            if definition.lower() == name:
                issues.append(f"{rid}: definition_long repeats the name")
            elif len(definition.split()) < profile.get("min_definition_words", 12):
                issues.append(f"{rid}: definition_long has fewer than {profile.get('min_definition_words', 12)} words")
        unit = (sd.get("measurement_unit") or "").strip().lower()
        if unit in {"n/a", "na", "none", "-"}:
            issues.append(f"{rid}: measurement_unit is a placeholder ({unit!r})")
    for rid, n in seen.items():
        if n > 1:
            issues.append(f"{rid}: idno appears {n} times")
    return issues


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("records", help="CSV export (flat) or JSON records (nested)")
    parser.add_argument("profile", help="completeness profile JSON")
    parser.add_argument("--schema", help="local copy of timeseries-schema.json")
    parser.add_argument("--level", default=None, help="profile level to check up to (default: the first level)")
    args = parser.parse_args()

    records = load_records(args.records)
    profile = json.load(open(args.profile, encoding="utf-8"))
    level = args.level or list(profile["levels"])[0]
    if level not in profile["levels"]:
        sys.exit(f"unknown level {level!r}; profile has {list(profile['levels'])}")
    print(f"{len(records)} records from {args.records}\n")

    # Layer 1
    valid, problems = layer_structure(records, load_schema(args.schema))
    print("Layer 1  Structure (World Bank timeseries-schema.json)")
    if valid is None:
        print(f"         {problems[0]}")
    else:
        print(f"         {valid} of {len(records)} records are valid instances of the schema")
        for p in problems[:20]:
            print(f"         {p}")
    print("         Note: the schema requires only idno and name. Validity is not completeness.\n")

    # Layer 2
    wanted, missing_by_record, missing_count = layer_completeness(records, profile, level)
    complete = len(records) - len(missing_by_record)
    print(f"Layer 2  Completeness against profile {profile.get('name', '')!r}, level {level!r} ({len(wanted)} fields)")
    print(f"         {complete} of {len(records)} records complete")
    for field, n in missing_count.most_common():
        print(f"         missing {field:<24} in {n} record(s)")
    for rid, missing in missing_by_record.items():
        print(f"         {rid}: {', '.join(missing)}")
    print()

    # Layer 3
    issues = layer_validity(records, profile)
    print(f"Layer 3  Validity: {len(issues)} issue(s)")
    for i in issues:
        print(f"         {i}")
    print()
    print("Not checked here: whether definitions are correct, whether name and definition agree,")
    print("whether the unit matches the values, or whether the text is specific. Those need review")
    print("or a semantic check (see Generative AI for Metadata Quality).")

    failed = bool(missing_by_record) or bool(issues) or (valid is not None and valid < len(records))
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
