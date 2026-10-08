"""Check a synthetic household-person pair of tables for structural validity.

Reads the household table (CSV: hhid, region, size) and the person table
(CSV: hhid, pid, relationship 1=head 2=spouse 3=child 4=other, age, sex)
of a synthetic release, and optionally the real pair, and reports:

    integrity   persons whose household does not exist; households with
                no persons; households whose size differs from their
                person count
    structure   households with no head or more than one; children older
                than the head; spouses more than 25 years apart from the head
    distributions  household size distribution and persons per household
                compared with the real pair when given (total variation
                distance)

Exit status 1 when any integrity or structure fault is found. Standard
library only.

Usage:
    python relational_check.py households_synth.csv persons_synth.csv \
        --real-households households_real.csv --real-persons persons_real.csv

What this does not do: the rules here are the ones every household
survey has; the organization adds its own (age at marriage, school age,
labour force status by age). A relational synthesizer that models the
parent-child structure (REaLTabFormer, HMA) usually passes integrity and
fails some structure rules, which is what the rules are for.
"""

from __future__ import annotations

import argparse
import csv
import sys
from collections import Counter, defaultdict
from pathlib import Path


def load(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


def tvd(a: Counter, b: Counter) -> float:
    na, nb = sum(a.values()), sum(b.values())
    return 0.5 * sum(abs(a[k] / na - b[k] / nb) for k in set(a) | set(b))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("households", type=Path)
    parser.add_argument("persons", type=Path)
    parser.add_argument("--real-households", type=Path)
    parser.add_argument("--real-persons", type=Path)
    args = parser.parse_args(argv)
    hh, pp = load(args.households), load(args.persons)
    by_hh: dict[str, list[dict[str, str]]] = defaultdict(list)
    for p in pp:
        by_hh[p["hhid"]].append(p)
    ids = {h["hhid"] for h in hh}

    faults = []
    orphans = [p for p in pp if p["hhid"] not in ids]
    faults += [
        f"person {p['hhid']}/{p['pid']} belongs to no household" for p in orphans
    ]
    faults += [
        f"household {h['hhid']} has no persons" for h in hh if h["hhid"] not in by_hh
    ]
    faults += [
        f"household {h['hhid']} size {h['size']} but {len(by_hh[h['hhid']])} persons"
        for h in hh
        if h["hhid"] in by_hh and int(h["size"]) != len(by_hh[h["hhid"]])
    ]
    for hid, members in by_hh.items():
        heads = [m for m in members if m["relationship"] == "1"]
        if len(heads) != 1:
            faults.append(f"household {hid} has {len(heads)} head(s)")
            continue
        head_age = int(heads[0]["age"])
        for m in members:
            if m["relationship"] == "3" and int(m["age"]) >= head_age:
                faults.append(
                    f"household {hid}: child {m['pid']} aged {m['age']} is not younger than the head ({head_age})"
                )
            if m["relationship"] == "2" and abs(int(m["age"]) - head_age) > 25:
                faults.append(
                    f"household {hid}: spouse {m['pid']} and head differ by more than 25 years"
                )

    print(f"{len(hh)} households, {len(pp)} persons")
    if faults:
        print(f"\n{len(faults)} fault(s):")
        for f in faults:
            print(f"  {f}")
    else:
        print("\nno integrity or structure faults")
    if args.real_households and args.real_persons:
        rhh, rpp = load(args.real_households), load(args.real_persons)
        rby = Counter(p["hhid"] for p in rpp)
        sizes_real = Counter(h["size"] for h in rhh)
        sizes_synth = Counter(h["size"] for h in hh)
        persons_real = Counter(str(n) for n in rby.values())
        persons_synth = Counter(str(len(m)) for m in by_hh.values())
        print(
            f"\nhousehold size distribution: TVD {tvd(sizes_real, sizes_synth):.3f}; persons per household: TVD {tvd(persons_real, persons_synth):.3f}"
        )
    return 1 if faults else 0


if __name__ == "__main__":
    sys.exit(main())
