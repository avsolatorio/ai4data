"""Check a server listing (server.json) before publishing it to a registry.

Reads a server.json in the MCP registry format and reports the required
fields (a reverse-DNS name such as org.example.stats/server, a semantic
version), the fields a client needs to connect (a remote with transport
streamable-http or sse and an https URL, or a package with a registry type
and a transport), and the fields a person needs to trust the listing
(title, description, website, repository). Exit status 1 when a required
or connection field is missing. Standard library only.

Usage:
    python check_server_listing.py server_listing.json

What this does not do: it checks the listing's shape against the format
documented by the registry at the time of writing; the registry validates
against its own schema (the ``$schema`` URL) on submission, and the
version of that schema changes. Whether the endpoint answers is the
operation chapter's check.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

NAME = re.compile(r"^[a-z0-9][a-z0-9.-]*\.[a-z0-9.-]+/[A-Za-z0-9._-]+$")
SEMVER = re.compile(r"^\d+\.\d+\.\d+(?:[-+][0-9A-Za-z.-]+)?$")
TRANSPORTS = {"streamable-http", "sse"}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("listing", type=Path)
    args = parser.parse_args(argv)
    with args.listing.open(encoding="utf-8") as fh:
        s = json.load(fh)
    errors: list[str] = []
    warnings: list[str] = []

    if not NAME.match(s.get("name", "")):
        errors.append(
            "name must be reverse-DNS namespace plus server name, for example org.example.stats/statistics-example"
        )
    if not SEMVER.match(str(s.get("version", ""))):
        errors.append("version must be a semantic version such as 1.0.0")
    remotes = s.get("remotes", [])
    packages = s.get("packages", [])
    if not remotes and not packages:
        errors.append(
            "a listing needs at least one remote (hosted endpoint) or one package (installable server)"
        )
    for i, r in enumerate(remotes):
        if r.get("type") not in TRANSPORTS:
            errors.append(f"remotes[{i}].type must be one of {sorted(TRANSPORTS)}")
        if not str(r.get("url", "")).startswith("https://"):
            errors.append(f"remotes[{i}].url must be an https URL")
    for i, p in enumerate(packages):
        for key in ("registryType", "identifier", "version", "transport"):
            if key not in p:
                errors.append(f"packages[{i}] lacks {key}")
    for key in ("title", "description", "websiteUrl"):
        if not s.get(key):
            warnings.append(f"{key} missing; clients show it to users")
    if not (s.get("repository") or {}).get("url"):
        warnings.append(
            "repository.url missing; readers cannot inspect the server's code"
        )
    if "$schema" not in s:
        warnings.append(
            "$schema missing; the registry validates against a dated schema"
        )
    desc = s.get("description", "")
    if desc and len(desc) > 300:
        warnings.append(f"description is {len(desc)} characters; keep it under 300")

    print(
        f"{args.listing.name}: {s.get('name')} {s.get('version')}; {len(remotes)} remote(s), {len(packages)} package(s)"
    )
    for e in errors:
        print(f"error    {e}")
    for w in warnings:
        print(f"warning  {w}")
    print(f"{len(errors)} error(s), {len(warnings)} warning(s)")
    return 1 if errors else 0


if __name__ == "__main__":
    sys.exit(main())
