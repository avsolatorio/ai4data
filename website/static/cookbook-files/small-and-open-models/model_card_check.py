"""Check a model card for what an organization needs before it deploys.

Reads a model card (JSON) and reports whether it states: the identifier,
version, and a hash of the weights; a licence with a URL; the languages;
the intended and out-of-scope uses; the training data (what, from where,
and whether confidential data were used); an evaluation with a suite
name, a date, and at least one number; known limitations; hardware
requirements; a contact. Exit status 1 when any is missing. Standard
library only.

Usage:
    python model_card_check.py model_card_example.json

What this does not do: it checks that the card says these things, not
that they are true. The evaluation numbers are re-produced on the
organization's own suite before deployment, and the licence is read by
someone who can say what it permits.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

REQUIRED = [
    "model_id",
    "version",
    "weights_sha256",
    "license",
    "license_url",
    "languages",
    "intended_use",
    "out_of_scope_use",
    "training_data",
    "evaluation",
    "known_limitations",
    "hardware",
    "contact",
]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("card", type=Path)
    args = parser.parse_args(argv)
    with args.card.open(encoding="utf-8") as fh:
        card = json.load(fh)
    problems = []
    for key in REQUIRED:
        value = card.get(key)
        if value in (None, "", [], {}):
            problems.append(f"{key} missing")
    if card.get("weights_sha256") and not re.fullmatch(
        r"[0-9a-f]{64}", card["weights_sha256"]
    ):
        problems.append("weights_sha256 is not a 64-character hex digest")
    ev = card.get("evaluation") or {}
    if ev and not (
        ev.get("suite")
        and ev.get("date")
        and any(isinstance(v, (int, float)) for v in ev.values())
    ):
        problems.append("evaluation lacks a suite name, a date, or a number")
    if card.get("training_data") and not re.search(
        r"\b(no external|left the organization|public|licensed|consent)\b",
        card["training_data"],
        re.IGNORECASE,
    ):
        problems.append(
            "training_data does not say whether confidential or external data were used"
        )
    print(
        f"{card.get('model_id', '?')} {card.get('version', '?')}: licence {card.get('license', '?')}, languages {', '.join(card.get('languages', []))}"
    )
    for key in REQUIRED:
        print(
            f"  {'ok     ' if not any(p.startswith(key) for p in problems) else 'MISSING'} {key}"
        )
    for p in problems:
        if not any(p.startswith(k + " missing") for k in REQUIRED):
            print(f"  problem {p}")
    print("result: " + ("FAIL" if problems else "PASS"))
    return 1 if problems else 0


if __name__ == "__main__":
    sys.exit(main())
