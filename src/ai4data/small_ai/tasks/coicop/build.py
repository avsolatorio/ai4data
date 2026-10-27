"""Build the label file and the real reference files of the COICOP task.

The task uses the classes of COICOP 1999 for household consumption (divisions 01 to 12), the version
that the European HICP and the UK CPI code their items to (ECOICOP is a finer split of the same classes).

Sources, downloaded by the operator (they are not redistributed here):

    UNSD, correspondence between COICOP 2018 and COICOP 1999, whose sheet "COICOP 1999" holds the
        codes, titles, and explanatory notes
        https://unstats.un.org/unsd/classifications/Econ/Download/COICOP2018_COICOP1999_correspondence_table_final.xlsx
    ONS, consumer price inflation item indices and price quotes (Open Government Licence v3.0)
        https://www.ons.gov.uk/economy/inflationandpriceindices/datasets/consumerpriceindicescpiandretailpricesindexrpiitemindicesandpricequotes
        - item indices of a month (ITEM_ID, ITEM_DESC), any number of months
        - the CPI classification framework (item_id, coicop4_id, start_date, end_date)
        - consumption segments of a month (CS_DESC, COICOP4_ID), any number of months

ONS codes a class as a number (10101 for 01.1.1) in the framework and as CP0111 in the segment files.
The reference set holds every distinct description that ONS assigned to exactly one class.

    python build.py --unsd CORRESPONDENCE.xlsx --framework FRAMEWORK.csv \
        --items itemindices202412.csv itemindices201908.csv --segments consumptionsegments202609.csv
"""

from __future__ import annotations

import argparse
import hashlib
import re
from pathlib import Path

import pandas as pd

HERE = Path(__file__).resolve().parent


def code_from_number(value) -> str | None:
    digits = re.sub(r"\D", "", str(value))
    if len(digits) < 5:
        return None
    return f"{int(digits[:-4]):02d}.{int(digits[-4:-2])}.{int(digits[-2:])}"


def code_from_cp(value) -> str | None:
    m = re.match(r"^CP(\d{2})(\d)(\d)$", str(value).strip())
    return f"{m[1]}.{m[2]}.{m[3]}" if m else None


def clean(text) -> str:
    text = str(text).replace("_x000D_", " ")
    return re.sub(r"\s+", " ", text).strip()


def labels(unsd: Path) -> pd.DataFrame:
    table = pd.read_excel(unsd, sheet_name="COICOP 1999", dtype=str)
    rows = []
    for r in table.to_dict("records"):
        code = str(r["Code"]).strip()
        if not re.match(r"^(0[1-9]|1[0-2])(\.\d(\.\d)?)?$", code):
            continue
        name = re.sub(r"\s*\((ND|SD|D|S)\)\s*$", "", clean(r["Description"]))
        note = clean(r.get("ExplanatoryNote") or "")
        note = "" if note == "nan" else note
        excludes = ""
        if "Excludes:" in note:
            note, excludes = note.split("Excludes:", 1)
        rows.append(
            {
                "id": code,
                "name": name,
                "parent": code.rsplit(".", 1)[0] if "." in code else "",
                "description": note.strip()[:700],
                "includes": "",
                "excludes": excludes.strip()[:300],
                "aliases": "",
            }
        )
    return pd.DataFrame(rows).sort_values("id")


def read_any(path: Path) -> pd.DataFrame:
    try:
        frame = pd.read_csv(path, dtype=str)
    except (UnicodeDecodeError, pd.errors.ParserError):
        frame = pd.read_excel(
            path, dtype=str
        )  # ONS publishes some item files as xlsx with a .csv name
    frame.columns = [c.strip() for c in frame.columns]
    return frame.apply(lambda s: s.str.strip() if s.dtype == object else s)


def reference(
    framework: Path, items: list[Path], segments: list[Path], leaves: set[str]
) -> pd.DataFrame:
    fw = read_any(framework)
    pairs = []
    for path in items:
        table = read_any(path)
        month = int(table["INDEX_DATE"].iloc[0])
        valid = fw[
            (fw["start_date"].astype(int) <= month)
            & (fw["end_date"].astype(int) >= month)
        ]
        merged = table.merge(valid, left_on="ITEM_ID", right_on="item_id")
        pairs += [
            (clean(d), code_from_number(c), f"item {path.name}")
            for d, c in zip(merged["ITEM_DESC"], merged["coicop4_id"])
        ]
    for path in segments:
        table = read_any(path).dropna(subset=["COICOP4_ID"])
        pairs += [
            (clean(d), code_from_cp(c), f"segment {path.name}")
            for d, c in zip(table["CS_DESC"], table["COICOP4_ID"])
        ]
    frame = pd.DataFrame(pairs, columns=["text", "code", "source"]).dropna()
    frame["key"] = (
        frame["text"]
        .str.upper()
        .str.replace(r"[^A-Z0-9]+", " ", regex=True)
        .str.strip()
    )
    codes = frame.groupby("key")["code"].agg(lambda s: sorted(set(s)))
    keep = codes[codes.map(len) == 1].map(lambda s: s[0])
    keep = keep[keep.isin(leaves)]
    first = frame.drop_duplicates("key").set_index("key")
    out = pd.DataFrame(
        {
            "text": first.loc[keep.index, "text"].values,
            "labels": keep.map(lambda c: [c]).values,
            "source": first.loc[keep.index, "source"].values,
        }
    )
    out.insert(
        0, "id", [hashlib.sha256(k.encode()).hexdigest()[:12] for k in keep.index]
    )
    return out


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--unsd", required=True)
    parser.add_argument("--framework", required=True)
    parser.add_argument("--items", nargs="+", default=[])
    parser.add_argument("--segments", nargs="+", default=[])
    parser.add_argument("--report-share", type=float, default=0.3)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args(argv)
    data = HERE / "data"
    data.mkdir(exist_ok=True)
    table = labels(Path(args.unsd))
    table.to_csv(data / "labels.csv", index=False)
    leaves = set(table["id"]) - set(table["parent"])
    ref = reference(
        Path(args.framework),
        [Path(p) for p in args.items],
        [Path(p) for p in args.segments],
        leaves,
    )
    ref = ref.sample(frac=1.0, random_state=args.seed)
    n_report = round(args.report_share * len(ref))
    ref.iloc[n_report:].to_json(
        data / "calibration.jsonl", orient="records", lines=True, force_ascii=False
    )
    ref.iloc[:n_report].to_json(
        data / "report.jsonl", orient="records", lines=True, force_ascii=False
    )
    measured = {c for ls in ref["labels"] for c in ls}
    print(f"labels {len(table)}  classes {len(leaves)}")
    print(
        f"descriptions {len(ref)}  calibration {len(ref) - n_report}  report {n_report}  classes in reference {len(measured)}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
