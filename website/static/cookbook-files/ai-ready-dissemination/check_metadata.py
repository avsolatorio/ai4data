"""Check catalog metadata in three layers, for any World Bank metadata type.

1. Structure: each record is validated against the World Bank metadata
   schema for its type (worldbank/metadata-schemas): indicator, indicators
   database, microdata (DDI Codebook), geospatial (ISO 19115/19139),
   document (Dublin Core), table, script, image, or video. The schemas
   require very little (an identifier and a title), so passing this layer
   means the record is well-formed, and nothing more.
2. Completeness: each record is checked against a profile, a JSON file in
   which the office lists the fields it requires at each maturity level,
   as dotted paths into the record. The schema does not decide what is
   complete; the office does.
3. Validity: rules named in the profile: dates parse, periods are ordered,
   URLs look like URLs, values come from a vocabulary, a text field has a
   minimum length, two fields are not identical, placeholders such as
   "n/a" are absent.

What this does not check: whether the text is correct, whether a title and
an abstract describe the same thing, whether a unit matches the values, or
whether the text is specific enough to be useful. Those are semantic
checks; see the program's Generative AI for Metadata Quality work.

Input: a JSON file with one record or a list of records in the nested form
of the schema (what NADA and the Metadata Editor export), or, for flat
types such as indicators, a CSV with one row per record. CSV columns are
dotted paths (or bare names placed under the profile's "csv_root"); list
fields take ";" between values and each value fills the item's main
property, which is read from the schema.

Usage:
    python check_metadata.py example_catalog.csv profile_indicator.json
    python check_metadata.py microdata_record.json profile_microdata.json
    python check_metadata.py records.json profile_geospatial.json --level ai-ready
    python check_metadata.py records.json profile.json --schema-dir ./wb-schemas

Needs the `jsonschema` package (4.18 or later, which brings `referencing`)
for layer 1. Without it, layer 1 is skipped and the report says so. The
schema files are downloaded once into a folder next to this script.
"""

import argparse
import csv
import json
import re
import sys
import urllib.request
from collections import Counter
from pathlib import Path

RAW = "https://raw.githubusercontent.com/worldbank/metadata-schemas/main/schemas/"
TYPES = {
    "indicator": "timeseries-schema.json",
    "indicators-db": "timeseries-db-schema.json",
    "microdata": "microdata-schema.json",
    "geospatial": "geospatial-schema.json",
    "document": "document-schema.json",
    "table": "table-schema.json",
    "script": "script-schema.json",
    "image": "image-schema.json",
    "video": "video-schema.json",
}
# Every file in the schemas folder; the type schemas reference the others.
SCHEMA_FILES = [
    "datacite-schema.json", "datafile-schema.json", "dcmi-schema.json", "ddi-schema.json",
    "document-schema.json", "geospatial-schema.json", "image-schema.json",
    "iptc-phovidmdshared-schema.json", "iptc-pmd-schema.json", "microdata-schema.json",
    "provenance-schema.json", "resource-schema.json", "script-schema.json", "table-schema.json",
    "timeseries-db-schema.json", "timeseries-schema.json", "variable-group-schema.json",
    "variable-schema.json", "video-schema.json",
]

DATE = re.compile(r"^\d{4}-\d{2}-\d{2}$")
PERIOD = re.compile(r"^\d{4}(-\d{2}(-\d{2})?|-Q[1-4]|-W\d{2}|-S[12])?$")
URL = re.compile(r"^https?://\S+$")
IDNO = re.compile(r"^[A-Za-z0-9_.\-]+$")
PLACEHOLDERS = {"n/a", "na", "none", "-", "tbd", "unknown", "?"}


# --- schemas ---------------------------------------------------------------

def ensure_schemas(schema_dir):
    """Download any missing schema file into schema_dir. Returns the files found."""
    schema_dir.mkdir(parents=True, exist_ok=True)
    found = {}
    for name in SCHEMA_FILES:
        path = schema_dir / name
        if not path.exists():
            try:
                with urllib.request.urlopen(RAW + name, timeout=20) as resp:
                    path.write_bytes(resp.read())
            except Exception as exc:  # noqa: BLE001
                print(f"could not download {name}: {exc}")
                continue
        found[name] = json.load(open(path, encoding="utf-8"))
    return found


def deref(schemas, name, node):
    """Follow $ref chains, across files, by file basename."""
    while isinstance(node, dict) and "$ref" in node:
        file, _, frag = node["$ref"].partition("#")
        if file:
            name = file.rsplit("/", 1)[-1]
        node = schemas[name]
        for part in [p for p in frag.split("/") if p]:
            node = node[part]
    return name, node


def child(schemas, name, node, key):
    """Return (name, node) of property `key`, looking through allOf."""
    name, node = deref(schemas, name, node)
    if key in node.get("properties", {}):
        return deref(schemas, name, node["properties"][key])
    for sub in node.get("allOf", []):
        found = child(schemas, name, sub, key)
        if found:
            return found
    return None


def node_at(schemas, root_name, path):
    """Schema node for a dotted path, stepping into array items."""
    name, node = deref(schemas, root_name, schemas[root_name])
    for key in path.split("."):
        if node.get("type") == "array":
            name, node = deref(schemas, name, node.get("items", {}))
        found = child(schemas, name, node, key)
        if not found:
            return None
        name, node = found
    return (name, node)


def item_key(schemas, root_name, path):
    """For an array-of-objects field, the property that a bare value fills."""
    found = node_at(schemas, root_name, path)
    if not found:
        return None
    name, node = found
    if node.get("type") != "array":
        return None
    name, item = deref(schemas, name, node.get("items", {}))
    if item.get("type") == "object" or "properties" in item:
        req = item.get("required") or []
        props = list(item.get("properties", {}))
        return (req or props or [None])[0]
    return ""  # array of plain values


# --- records ---------------------------------------------------------------

def set_path(record, path, value, schemas, root_name):
    """Set a dotted path, creating the first item of any array on the way."""
    parts = path.split(".")
    node = record
    for i, key in enumerate(parts[:-1]):
        sub = ".".join(parts[: i + 1])
        found = node_at(schemas, root_name, sub) if schemas else None
        is_array = bool(found) and found[1].get("type") == "array"
        if is_array:
            node.setdefault(key, [{}])
            node = node[key][0]
        else:
            node = node.setdefault(key, {})
    node[parts[-1]] = value


def load_records(path, profile, schemas, root_name):
    if path.endswith(".json"):
        data = json.load(open(path, encoding="utf-8"))
        return data if isinstance(data, list) else [data]
    root = profile.get("csv_root", "")
    renames = profile.get("csv_columns", {})
    records = []
    with open(path, newline="", encoding="utf-8") as fh:
        for row in csv.DictReader(fh):
            rec = {}
            for col, raw in row.items():
                val = (raw or "").strip()
                if not val:
                    continue
                target = renames.get(col, col)
                if root and not target.startswith(root + "."):
                    target = f"{root}.{target}"
                key = item_key(schemas, root_name, target) if schemas else None
                if key:
                    value = [{key: v.strip()} for v in val.split(";") if v.strip()]
                elif key == "":
                    value = [v.strip() for v in val.split(";") if v.strip()]
                else:
                    value = val
                set_path(rec, target, value, schemas, root_name)
            records.append(rec)
    return records


def values_at(obj, parts):
    """All values at a dotted path, flattening through lists."""
    if not parts:
        return [obj]
    if isinstance(obj, list):
        return [v for item in obj for v in values_at(item, parts)]
    if isinstance(obj, dict) and parts[0] in obj:
        return values_at(obj[parts[0]], parts[1:])
    return []


def nonempty(v):
    if v is None:
        return False
    if isinstance(v, str):
        return bool(v.strip())
    if isinstance(v, (list, dict)):
        return len(v) > 0
    return True


def present(record, path):
    return any(nonempty(v) for v in values_at(record, path.split(".")))


def first(record, path):
    vals = [v for v in values_at(record, path.split(".")) if nonempty(v)]
    return vals[0] if vals else None


def strings_at(record, path):
    return [v for v in values_at(record, path.split(".")) if isinstance(v, str) and v.strip()]


# --- layers ----------------------------------------------------------------

def layer_structure(records, schemas, root_name, id_path):
    try:
        import jsonschema
        from referencing import Registry, Resource
        from referencing.jsonschema import DRAFT202012
    except ImportError:
        return None, ["jsonschema 4.18+ not installed; layer 1 skipped"]
    if root_name not in schemas:
        return None, ["schema not available; layer 1 skipped"]

    def strip(s):
        s = dict(s)
        s.pop("$schema", None)  # the files name a metaschema jsonschema does not know
        return s

    def retrieve(uri):
        name = uri.rsplit("/", 1)[-1]
        if name in schemas:
            return Resource.from_contents(strip(schemas[name]), default_specification=DRAFT202012)
        from referencing.exceptions import NoSuchResource
        raise NoSuchResource(ref=uri)

    registry = Registry(retrieve=retrieve)
    validator = jsonschema.Draft202012Validator(strip(schemas[root_name]), registry=registry)
    problems = []
    invalid = set()
    for i, rec in enumerate(records):
        rid = first(rec, id_path) or f"record {i + 1}"
        try:
            for err in validator.iter_errors(rec):
                where = "/".join(str(p) for p in err.absolute_path)
                problems.append(f"{rid}: {where}: {err.message[:90]}")
                invalid.add(rid)
        except Exception as exc:  # noqa: BLE001
            problems.append(f"{rid}: validation failed: {exc}")
            invalid.add(rid)
    return len(records) - len(invalid), problems


def layer_completeness(records, profile, level, id_path):
    levels = profile["levels"]
    order = list(levels)
    wanted = []
    for lvl in order[: order.index(level) + 1]:
        wanted += levels[lvl]
    missing_by_record = {}
    missing_count = Counter()
    for i, rec in enumerate(records):
        rid = first(rec, id_path) or f"record {i + 1}"
        missing = [f for f in wanted if not present(rec, f)]
        if missing:
            missing_by_record[rid] = missing
            missing_count.update(missing)
    return wanted, missing_by_record, missing_count


def layer_validity(records, profile, id_path):
    rules = profile.get("rules", {})
    issues = []
    seen = Counter()
    for i, rec in enumerate(records):
        rid = first(rec, id_path) or f"record {i + 1}"
        seen[rid] += 1
        if first(rec, id_path) and not IDNO.match(str(first(rec, id_path))):
            issues.append(f"{rid}: identifier has characters outside A-Z, 0-9, _ . -")
        for path in rules.get("dates", []):
            for v in strings_at(rec, path):
                if not DATE.match(v):
                    issues.append(f"{rid}: {path} {v!r} is not YYYY-MM-DD")
        for path in rules.get("periods", []):
            for v in strings_at(rec, path):
                if not PERIOD.match(v):
                    issues.append(f"{rid}: {path} {v!r} is not an ISO 8601 or SDMX period")
        for start_path, end_path in rules.get("period_order", []):
            s, e = first(rec, start_path), first(rec, end_path)
            if isinstance(s, str) and isinstance(e, str) and PERIOD.match(s) and PERIOD.match(e) and e < s:
                issues.append(f"{rid}: {end_path} ({e}) is before {start_path} ({s})")
        for path in rules.get("urls", []):
            for v in strings_at(rec, path):
                if not URL.match(v):
                    issues.append(f"{rid}: {path} {v!r} is not a URL")
        for path, allowed in rules.get("vocabularies", {}).items():
            for v in strings_at(rec, path):
                if v not in allowed:
                    issues.append(f"{rid}: {path} {v!r} is not in the profile vocabulary")
        for path, n in rules.get("min_words", {}).items():
            for v in strings_at(rec, path):
                if len(v.split()) < n:
                    issues.append(f"{rid}: {path} has fewer than {n} words")
        for a, b in rules.get("not_equal", []):
            va, vb = first(rec, a), first(rec, b)
            if isinstance(va, str) and isinstance(vb, str) and va.strip().lower() == vb.strip().lower():
                issues.append(f"{rid}: {b} repeats {a}")
        for path in rules.get("placeholders", []):
            for v in strings_at(rec, path):
                if v.strip().lower() in PLACEHOLDERS:
                    issues.append(f"{rid}: {path} is a placeholder ({v!r})")
    for rid, n in seen.items():
        if n > 1:
            issues.append(f"{rid}: identifier appears {n} times")
    return issues


# --- main ------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("records", help="JSON records (nested) or CSV (flat)")
    parser.add_argument("profile", help="completeness profile JSON for the data type")
    parser.add_argument("--type", help="data type; default: the profile's \"type\"", choices=sorted(TYPES))
    parser.add_argument("--level", help="profile level to check up to; default: the first level")
    parser.add_argument("--schema-dir", default=str(Path(__file__).with_name("wb-schemas")),
                        help="folder holding the World Bank schema files (downloaded if missing)")
    args = parser.parse_args()

    profile = json.load(open(args.profile, encoding="utf-8"))
    dtype = args.type or profile.get("type")
    if dtype not in TYPES:
        sys.exit(f"unknown data type {dtype!r}; choose from {sorted(TYPES)}")
    root_name = TYPES[dtype]
    id_path = profile.get("rules", {}).get("id", "idno")
    level = args.level or list(profile["levels"])[0]
    if level not in profile["levels"]:
        sys.exit(f"unknown level {level!r}; profile has {list(profile['levels'])}")

    schemas = ensure_schemas(Path(args.schema_dir))
    records = load_records(args.records, profile, schemas, root_name)
    print(f"{len(records)} {dtype} record(s) from {args.records}\n")

    valid, problems = layer_structure(records, schemas, root_name, id_path)
    print(f"Layer 1  Structure (World Bank {root_name})")
    if valid is None:
        print(f"         {problems[0]}")
    else:
        print(f"         {valid} of {len(records)} records are valid instances of the schema")
        for p in problems[:20]:
            print(f"         {p}")
    print("         Note: the schemas require little more than an identifier and a title. Validity is not completeness.\n")

    wanted, missing_by_record, missing_count = layer_completeness(records, profile, level, id_path)
    complete = len(records) - len(missing_by_record)
    print(f"Layer 2  Completeness against profile {profile.get('name', '')!r}, level {level!r} ({len(wanted)} fields)")
    print(f"         {complete} of {len(records)} records complete")
    reasons = profile.get("fields", {})
    for field, n in missing_count.most_common():
        print(f"         missing {field:<48} in {n} record(s)")
        if field in reasons:
            print(f"                 why: {reasons[field]}")
    for rid, missing in missing_by_record.items():
        print(f"         {rid}: {', '.join(missing)}")
    print()

    issues = layer_validity(records, profile, id_path)
    print(f"Layer 3  Validity: {len(issues)} issue(s)")
    for i in issues:
        print(f"         {i}")
    print()
    print("Not checked here: whether the text is correct, whether title and abstract agree, whether a unit")
    print("matches the values, or whether the text is specific. Those need review or a semantic check")
    print("(see Generative AI for Metadata Quality).")

    failed = bool(missing_by_record) or bool(issues) or (valid is not None and valid < len(records))
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
