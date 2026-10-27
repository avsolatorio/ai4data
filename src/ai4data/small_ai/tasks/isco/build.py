"""Build the label file and the real reference files of the ISCO-08 task.

Sources, downloaded by the operator (they are not redistributed here):

    ILO, ISCO-08 structure and definitions (English), from https://isco.ilo.org/en/isco-08/
        https://webapps.ilo.org/ilostat-files/ISCO/newdocs-08-2021/ISCO-08/ISCO-08%20EN%20Structure%20and%20definitions.xlsx
    tabiya/occupation_titles_esco on Hugging Face (CC BY 4.0): job titles from the Ethiopian job
        board hahu.jobs with the ESCO occupation the board assigned; the first four digits of an
        ESCO code are its ISCO-08 unit group
        https://huggingface.co/datasets/tabiya/occupation_titles_esco

The label file has the 619 groups of ISCO-08; the 436 unit groups are the labels the model predicts.
Each unit group carries its definition and its example occupations. The reference set keeps the titles
whose ESCO codes all fall in one unit group, and draws the calibration and report files from them.

    python build.py --ilo STRUCTURE.xlsx --hahu occupations_training_set.csv
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path

import pandas as pd

HERE = Path(__file__).resolve().parent


def items(cell, header: str) -> list[str]:
    if not isinstance(cell, str):
        return []
    lines = [line.strip(" -\t") for line in cell.replace(header, "").splitlines()]
    return [
        line for line in lines if line and not line.lower().startswith("examples of")
    ]


def labels(structure: Path) -> pd.DataFrame:
    table = pd.read_excel(structure, dtype=str)
    rows = []
    for r in table.to_dict("records"):
        code = str(r["ISCO 08 Code"]).strip()
        definition = re.sub(r"\s+", " ", str(r.get("Definition") or "")).strip()
        rows.append(
            {
                "id": code,
                "name": str(r["Title EN"]).strip(),
                "parent": code[:-1] if len(code) > 1 else "",
                "description": definition[:600] if definition != "nan" else "",
                "includes": "; ".join(
                    items(
                        r.get("Included occupations"),
                        "Examples of the occupations classified here:",
                    )[:12]
                ),
                "excludes": "; ".join(
                    items(
                        r.get("Excluded occupations"),
                        "Some related occupations classified elsewhere:",
                    )[:8]
                ),
                "aliases": "",
            }
        )
    return pd.DataFrame(rows).sort_values("id")


def reference(
    hahu: Path, leaves: set[str], calibration: int, report: int, seed: int
) -> dict[str, pd.DataFrame]:
    table = pd.read_csv(hahu, dtype=str)
    table["isco"] = table["esco_code"].str.extract(r"^(\d{4})")[0]
    codes = (
        table.dropna(subset=["isco"])
        .groupby("title")["isco"]
        .agg(lambda s: sorted(set(s)))
    )
    single = codes[codes.map(len) == 1].map(lambda s: s[0])
    single = single[single.isin(leaves)]
    frame = pd.DataFrame(
        {"text": single.index.str.strip(), "labels": single.map(lambda c: [c]).values}
    )
    frame = (
        frame[frame["text"].str.len() > 1]
        .drop_duplicates("text")
        .sample(frac=1.0, random_state=seed)
    )
    frame.insert(0, "id", [f"hahu-{i:05d}" for i in range(len(frame))])
    return {
        "calibration": frame.iloc[:calibration],
        "report": frame.iloc[calibration : calibration + report],
        "_stats": pd.DataFrame(
            [{"titles": len(codes), "one_unit_group": len(single), "kept": len(frame)}]
        ),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--ilo", required=True)
    parser.add_argument("--hahu", required=True)
    parser.add_argument("--calibration", type=int, default=3000)
    parser.add_argument("--report", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args(argv)
    data = HERE / "data"
    data.mkdir(exist_ok=True)
    table = labels(Path(args.ilo))
    table.to_csv(data / "labels.csv", index=False)
    parents = set(table["parent"])
    leaves = set(table["id"]) - parents
    parts = reference(Path(args.hahu), leaves, args.calibration, args.report, args.seed)
    for name in ("calibration", "report"):
        parts[name].to_json(
            data / f"{name}.jsonl", orient="records", lines=True, force_ascii=False
        )
    stats = parts["_stats"].iloc[0].to_dict()
    measured = {
        c
        for part in ("calibration", "report")
        for ls in parts[part]["labels"]
        for c in ls
    }
    print(f"labels {len(table)}  unit groups {len(leaves)}")
    print(
        f"titles {stats['titles']}  in one unit group {stats['one_unit_group']}  kept {stats['kept']}"
    )
    print(
        f"calibration {len(parts['calibration'])}  report {len(parts['report'])}  unit groups in reference {len(measured)}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
