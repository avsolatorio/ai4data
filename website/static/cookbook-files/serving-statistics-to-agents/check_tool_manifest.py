"""Check a tool manifest against the design rules for a statistics server.

Reads a manifest (JSON: server, version, description, tools, resources) and
checks each tool against the rules of the cookbook's tools chapter:

errors (exit status 1)
  - a tool name that is not a lower-case verb phrase in snake_case
  - a description shorter than 20 words (the description is the model's
    only documentation)
  - a tool that is not marked read-only (readOnlyHint true)
  - an input without a type or a description
  - a data tool (one that returns observations) whose outputs lack a
    provenance field: SERIES, UNIT_MEASURE, RELEASE, SOURCE_URL, license,
    citation
  - a tool without an example call
  - more than ten tools

warnings
  - no guidance resource (a resource whose description tells the model how
    to use the tools)
  - a search tool without a limit input

The manifest is a design document; it is not the server. The checks apply
to whatever the server's tool list reports as well, when exported in the
same form. Standard library only.

Usage:
    python check_tool_manifest.py tool_manifest.json
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

NAME = re.compile(r"^[a-z][a-z0-9]*(_[a-z0-9]+)+$")
PROVENANCE = ["SERIES", "UNIT_MEASURE", "RELEASE", "SOURCE_URL", "license", "citation"]
MAX_TOOLS = 10
MIN_WORDS = 20


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("manifest", type=Path)
    args = parser.parse_args(argv)
    with args.manifest.open(encoding="utf-8") as fh:
        m = json.load(fh)

    errors: list[str] = []
    warnings: list[str] = []
    tools = m.get("tools", [])
    if len(tools) > MAX_TOOLS:
        errors.append(
            f"{len(tools)} tools; more than {MAX_TOOLS} makes the interface hard for a model to use"
        )
    for t in tools:
        name = t.get("name", "?")
        if not NAME.match(name):
            errors.append(f"{name}: name must be a snake_case verb phrase (search_series, get_observations)")
        words = len((t.get("description") or "").split())
        if words < MIN_WORDS:
            errors.append(f"{name}: description has {words} words; the model needs at least {MIN_WORDS}")
        if t.get("readOnlyHint") is not True:
            errors.append(f"{name}: not marked read-only (readOnlyHint true)")
        for arg, spec in (t.get("inputs") or {}).items():
            if not spec.get("type") or not spec.get("description"):
                errors.append(f"{name}: input {arg!r} needs a type and a description")
        outputs = t.get("outputs") or []
        if "observations" in outputs:
            missing = [f for f in PROVENANCE if f not in outputs]
            if missing:
                errors.append(f"{name}: data tool lacks provenance fields {missing}")
        if not t.get("example"):
            errors.append(f"{name}: no example call")
        if name.startswith("search") and "limit" not in (t.get("inputs") or {}):
            warnings.append(f"{name}: a search tool without a limit input returns unbounded lists")
    guidance = any("how to use" in (r.get("description") or "").lower() for r in m.get("resources", []))
    if not guidance:
        warnings.append(
            "no guidance resource; a resource that tells the model how to use the tools reduces misuse"
        )

    print(
        f"{m.get('server', '?')} {m.get('version', '')}: {len(tools)} tool(s), {len(m.get('resources', []))} resource(s)"
    )
    for w in warnings:
        print(f"warning  {w}")
    for e in errors:
        print(f"error    {e}")
    print(f"\n{len(errors)} error(s), {len(warnings)} warning(s)")
    print(
        "Not checked here: whether the server behaves as the manifest says; the evaluation in chapter 5 does that."
    )
    return 1 if errors else 0


if __name__ == "__main__":
    sys.exit(main())
