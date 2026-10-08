"""Count views, downloads, and API calls per dataset the way the COUNTER
Code of Practice for Research Data counts them.

Reads an access log (CSV: timestamp, dataset_id, action view|download|api,
ip_hash, user_agent, status) and applies three rules before counting:

    robots       requests from known crawlers are excluded entirely
    double-click repeated requests by the same client for the same
                 dataset and action within 30 seconds count once
    status       only successful requests (status 2xx) count

Counts, per dataset and month, the regular requests (browsers), the
machine requests (API clients and scripted tools, reported separately as
the Code requires), and the unique investigations (distinct clients that
touched the dataset). Standard library only.

Usage:
    python count_access.py access_log.csv

What this does not do: a hashed IP is a weak client identifier (shared
networks, changing addresses); the Code allows a session cookie or a
user identifier where the catalog has one. The robot list here is short;
the Code points to a maintained list. Counts of files are not counts of
users or uses; the use report presents them beside the mention counts.
"""

from __future__ import annotations

import argparse
import csv
import sys
from collections import defaultdict
from datetime import datetime, timedelta
from pathlib import Path

ROBOTS = (
    "googlebot",
    "bingbot",
    "yandex",
    "baiduspider",
    "duckduckbot",
    "crawler",
    "spider",
    "slurp",
)
MACHINE = (
    "python-requests",
    "curl/",
    "wget/",
    "httpx",
    "java/",
    "go-http-client",
    "agent/",
)
DOUBLE_CLICK = timedelta(seconds=30)


def kind(user_agent: str, action: str) -> str:
    ua = user_agent.lower()
    if any(r in ua for r in ROBOTS):
        return "robot"
    if action == "api" or any(m in ua for m in MACHINE):
        return "machine"
    return "regular"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("log", type=Path)
    args = parser.parse_args(argv)
    with args.log.open(newline="", encoding="utf-8") as fh:
        rows = sorted(csv.DictReader(fh), key=lambda r: r["timestamp"])

    excluded = {"robot": 0, "double-click": 0, "status": 0}
    last_seen: dict[tuple[str, str, str], datetime] = {}
    counts: dict[tuple[str, str], dict[str, int]] = defaultdict(
        lambda: defaultdict(int)
    )
    clients: dict[tuple[str, str], set[str]] = defaultdict(set)
    for r in rows:
        if not r["status"].startswith("2"):
            excluded["status"] += 1
            continue
        k = kind(r["user_agent"], r["action"])
        if k == "robot":
            excluded["robot"] += 1
            continue
        when = datetime.fromisoformat(r["timestamp"].replace("Z", "+00:00"))
        key = (r["ip_hash"], r["dataset_id"], r["action"])
        if key in last_seen and when - last_seen[key] <= DOUBLE_CLICK:
            excluded["double-click"] += 1
            last_seen[key] = when
            continue
        last_seen[key] = when
        month = r["timestamp"][:7]
        counts[(r["dataset_id"], month)][f"{k}_{r['action']}"] += 1
        counts[(r["dataset_id"], month)][k] += 1
        clients[(r["dataset_id"], month)].add(r["ip_hash"])

    print(
        f"{len(rows)} log rows; excluded: robots {excluded['robot']}, double-clicks {excluded['double-click']}, failed requests {excluded['status']}\n"
    )
    print(
        f"{'dataset':<22} {'month':<8} {'regular':>8} {'views':>6} {'downl.':>6} {'machine':>8} {'api':>4} {'clients':>8}"
    )
    for (dataset, month), c in sorted(counts.items()):
        print(
            f"{dataset:<22} {month:<8} {c['regular']:>8} {c['regular_view']:>6} {c['regular_download']:>6} "
            f"{c['machine']:>8} {c['machine_api']:>4} {len(clients[(dataset, month)]):>8}"
        )
    print(
        "\nRegular and machine requests are reported separately; neither counts users or uses."
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
