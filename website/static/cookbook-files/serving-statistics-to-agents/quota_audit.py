"""Audit an agent interface's request log against the client registry.

Reads the client registry (CSV: client_id, name, tier, requests_per_minute,
daily_quota, terms_accepted, contact) and a request log (CSV: timestamp,
client_id, tool, status) and reports, per client, the requests in the
period, the busiest minute against the per-minute limit, the share of
requests the server rejected (status 429 or 401), and two policy checks:
clients in the log that the registry does not know, and registered
clients that have not accepted the terms. Standard library only.

Usage:
    python quota_audit.py client_registry.csv agent_requests.csv

Exit status 1 when an unknown client made requests or a registered client
without accepted terms did, so that the audit can run as a periodic check.

What this does not do: it audits a log after the fact. The limits
themselves are enforced by the server or a gateway in front of it; the
audit shows whether the limits, the tiers, and the registry match what
happens. Quotas protect the service; the terms protect the data's
attribution, and both are stated to clients at registration.
"""

from __future__ import annotations

import argparse
import csv
import sys
from collections import Counter, defaultdict
from pathlib import Path


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("registry", type=Path)
    parser.add_argument("log", type=Path)
    args = parser.parse_args(argv)
    with args.registry.open(newline="", encoding="utf-8") as fh:
        registry = {r["client_id"]: r for r in csv.DictReader(fh)}
    with args.log.open(newline="", encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh))

    per_client: dict[str, list[dict[str, str]]] = defaultdict(list)
    for r in rows:
        per_client[r["client_id"]].append(r)

    print(
        f"{len(rows)} requests from {len(per_client)} clients; registry lists {len(registry)}\n"
    )
    print(
        f"{'client':<10} {'tier':<11} {'requests':>8} {'busiest min':>11} {'limit/min':>9} {'rejected':>8}  note"
    )
    unknown: list[str] = []
    no_terms: list[str] = []
    for client, reqs in sorted(per_client.items(), key=lambda kv: -len(kv[1])):
        reg = registry.get(client)
        minutes = Counter(r["timestamp"][:16] for r in reqs)
        busiest = max(minutes.values())
        rejected = sum(1 for r in reqs if r["status"] in ("429", "401", "403"))
        if reg is None:
            unknown.append(client)
            print(
                f"{client:<10} {'unknown':<11} {len(reqs):>8} {busiest:>11} {'-':>9} {rejected:>8}  not in the registry"
            )
            continue
        limit = int(reg["requests_per_minute"])
        notes = []
        if busiest > limit:
            notes.append(f"busiest minute exceeds limit ({busiest} > {limit})")
        if reg["tier"] != "public" and not reg["terms_accepted"]:
            no_terms.append(client)
            notes.append("terms not accepted")
        print(
            f"{client:<10} {reg['tier']:<11} {len(reqs):>8} {busiest:>11} {limit:>9} {rejected:>8}  {'; '.join(notes)}"
        )

    print()
    if unknown:
        print(
            f"unknown clients: {', '.join(unknown)} (rejected by the server; check for a leaked or revoked key)"
        )
    if no_terms:
        print(
            f"registered without accepted terms: {', '.join(no_terms)} (suspend until the terms are accepted)"
        )
    tools = Counter(r["tool"] for r in rows if r["status"] == "200")
    print("tool calls served: " + ", ".join(f"{t} {n}" for t, n in tools.most_common()))
    return 1 if unknown or no_terms else 0


if __name__ == "__main__":
    sys.exit(main())
