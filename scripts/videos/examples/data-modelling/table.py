"""The worked example of the data-modelling guide: learners in Rwanda by sex and type of school.

Counts are transcribed from Table 4.1 as reproduced in "AI readiness of data
and metadata: A Practical Guide to Preparing Statistical Data Files for
SDMX-Compatible Dissemination" (World Bank, version 1.0, 2026-10-18, chapter
10; source of the table: School census, Ministry of Education). The script
checks that male and female add up to each total and that the three school
types add up to all schools, computes the sex distribution in percent from
the counts, and writes table.json for the video stage.
"""
import json
from pathlib import Path

YEARS = ["2017", "2018", "2019", "2020/21", "2021/22", "2022/23", "2023/24"]
COUNTS = {  # (school type, sex): counts by year
    ("_T", "_T"): [3614004, 3619871, 3750252, 4029848, 4159782, 4456419, 4766125],
    ("_T", "M"): [1783151, 1786491, 1854214, 1975737, 2050664, 2209396, 2364153],
    ("_T", "F"): [1830853, 1833380, 1896038, 2054111, 2109118, 2247023, 2401972],
    ("PUB", "_T"): [1098480, 1110454, 1171768, 1506280, 1590690, 1725647, 1879240],
    ("PUB", "M"): [546970, 556061, 588029, 745710, 792307, 862398, 936801],
    ("PUB", "F"): [551510, 554393, 583739, 760570, 798383, 863249, 942439],
    ("GSS", "_T"): [2149617, 2159576, 2189152, 2134012, 2169708, 2299765, 2411376],
    ("GSS", "M"): [1050577, 1056498, 1071594, 1036099, 1058085, 1129641, 1186399],
    ("GSS", "F"): [1099040, 1103078, 1117558, 1097913, 1111623, 1170124, 1224977],
    ("PRV", "_T"): [365907, 349841, 389332, 389556, 399384, 431007, 475509],
    ("PRV", "M"): [185604, 173932, 194591, 193928, 200272, 217357, 240953],
    ("PRV", "F"): [180303, 175909, 194741, 195628, 199112, 213650, 234556],
}


def main():
    problems = []
    for school in ("_T", "PUB", "GSS", "PRV"):
        for i, year in enumerate(YEARS):
            m, f, t = (COUNTS[(school, s)][i] for s in ("M", "F", "_T"))
            if m + f != t:
                problems.append(f"{school} {year}: male + female = {m + f}, total = {t}")
    for sex in ("_T", "M", "F"):
        for i, year in enumerate(YEARS):
            parts = sum(COUNTS[(s, sex)][i] for s in ("PUB", "GSS", "PRV"))
            if parts != COUNTS[("_T", sex)][i]:
                problems.append(f"sex {sex} {year}: school types = {parts}, all schools = {COUNTS[('_T', sex)][i]}")
    shares = {f"{school}|{sex}": [round(100 * COUNTS[(school, sex)][i] / COUNTS[(school, '_T')][i], 1) for i in range(len(YEARS))]
              for school in ("_T", "PUB", "GSS", "PRV") for sex in ("M", "F")}
    out = {"years": YEARS, "counts": {f"{k[0]}|{k[1]}": v for k, v in COUNTS.items()}, "shares": shares, "checks": problems}
    Path(__file__).with_name("table.json").write_text(json.dumps(out, indent=1) + "\n", encoding="utf-8")
    print("checks:", problems or "all sums agree")
    print("shares, all schools:", shares["_T|M"], shares["_T|F"])
    print("shares, public:", shares["PUB|M"], shares["PUB|F"])


if __name__ == "__main__":
    main()
