"""Geography resources for realistic searches: economies, groupings, subnational places, multilingual names.

  python -m relevance_explainer.synth.geo build      # writes data/geo/geo.json (network: World Bank API, GeoNames)

Contents of geo.json:
  countries  iso3 -> {iso2, name, region, income, capital, names: {language: [names]}}
  groups     code -> {name, kind, members: [iso3], names: {language: [names]}}   (World Bank regions, income groups,
             other World Bank aggregates, and a curated set of common groupings)
  places     list of {name, ascii, alt: [names], iso3, kind: admin1|city, admin1, population}
"""
from __future__ import annotations

import argparse
import io
import json
import urllib.request
import zipfile
from pathlib import Path

GEO = Path("data/geo/geo.json")
LANG_CODES = {"English": "en", "French": "fr", "Spanish": "es", "Portuguese": "pt", "Arabic": "ar", "Chinese": "zh",
              "Russian": "ru", "Indonesian": "id", "Hindi": "hi", "Turkish": "tr", "Swahili": "sw"}
UA = {"User-Agent": "curl/8.5.0"}

# Common groupings without a World Bank aggregate code. Membership as generally used; where definitions vary
# (Horn of Africa, Sahel) the narrower, most common definition is used.
CURATED = {
    "SAHEL_G5": ("Sahel", ["BFA", "TCD", "MLI", "MRT", "NER"]),
    "HORN": ("Horn of Africa", ["DJI", "ERI", "ETH", "SOM"]),
    "G7": ("G7", ["CAN", "FRA", "DEU", "ITA", "JPN", "GBR", "USA"]),
    "BRICS": ("BRICS", ["BRA", "RUS", "IND", "CHN", "ZAF"]),
    "ASEAN": ("ASEAN", ["BRN", "KHM", "IDN", "LAO", "MYS", "MMR", "PHL", "SGP", "THA", "VNM"]),
    "EAC": ("East African Community", ["BDI", "COD", "KEN", "RWA", "SOM", "SSD", "TZA", "UGA"]),
    "SADC": ("SADC", ["AGO", "BWA", "COM", "COD", "SWZ", "LSO", "MDG", "MWI", "MUS", "MOZ", "NAM", "SYC", "ZAF", "TZA", "ZMB", "ZWE"]),
    "ECOWAS": ("ECOWAS", ["BEN", "CPV", "CIV", "GMB", "GHA", "GIN", "GNB", "LBR", "NGA", "SEN", "SLE", "TGO"]),
    "CENTRAL_ASIA": ("Central Asia", ["KAZ", "KGZ", "TJK", "TKM", "UZB"]),
    "SOUTHEAST_ASIA": ("Southeast Asia", ["BRN", "KHM", "IDN", "LAO", "MYS", "MMR", "PHL", "SGP", "THA", "TLS", "VNM"]),
    "CARIBBEAN": ("Caribbean", ["ATG", "BHS", "BRB", "CUB", "DMA", "DOM", "GRD", "HTI", "JAM", "KNA", "LCA", "VCT", "TTO"]),
    "PACIFIC_ISLANDS": ("Pacific Islands", ["FJI", "KIR", "MHL", "FSM", "NRU", "PLW", "PNG", "WSM", "SLB", "TON", "TUV", "VUT"]),
    "MAGHREB": ("Maghreb", ["DZA", "LBY", "MRT", "MAR", "TUN"]),
    "GCC": ("Gulf Cooperation Council", ["BHR", "KWT", "OMN", "QAT", "SAU", "ARE"]),
    "BALKANS": ("Western Balkans", ["ALB", "BIH", "XKX", "MNE", "MKD", "SRB"]),
    "ANDEAN": ("Andean countries", ["BOL", "COL", "ECU", "PER"]),
    "CENTRAL_AMERICA": ("Central America", ["BLZ", "CRI", "SLV", "GTM", "HND", "NIC", "PAN"]),
    "LAKE_CHAD": ("Lake Chad Basin", ["CMR", "TCD", "NER", "NGA"]),
}


def _our_script(s: str) -> bool:
    """Latin, Cyrillic, Arabic, Devanagari or Han: the scripts of the 11 languages."""
    return all(c.isascii() or "\u00c0" <= c <= "\u024f" or "\u0400" <= c <= "\u04ff" or "\u0600" <= c <= "\u06ff"
               or "\u0900" <= c <= "\u097f" or "\u4e00" <= c <= "\u9fff" or c in " -'’." for c in s)


def _get(url: str) -> bytes:
    with urllib.request.urlopen(urllib.request.Request(url, headers=UA), timeout=60) as r:
        return r.read()


def _wb(path: str) -> list[dict]:
    data = json.loads(_get(f"https://api.worldbank.org/v2/{path}{'&' if '?' in path else '?'}format=json&per_page=500"))
    return data[1] if isinstance(data, list) and len(data) > 1 and data[1] else []


def build() -> None:
    from babel import Locale

    countries = {}
    for c in _wb("country"):
        if c["region"]["id"] == "NA":  # aggregates
            continue
        countries[c["id"]] = {"iso2": c["iso2Code"], "name": c["name"], "region": c["region"]["value"].strip(),
                              "income": c["incomeLevel"]["value"], "capital": c["capitalCity"], "names": {}}
    for lang, code in LANG_CODES.items():
        loc = Locale.parse(code)
        for iso3, c in countries.items():
            n = loc.territories.get(c["iso2"])
            names = {c["name"]} if lang == "English" else set()
            if n:
                names.add(n)
            c["names"][lang] = sorted(names)
    print(f"countries: {len(countries)}")

    groups = {}
    for r in _wb("region"):
        code = r.get("code") or r.get("id")
        members = [c["id"] for c in _wb(f"country?region={code}") if c["id"] in countries]
        if members:
            groups[code] = {"name": r["name"].strip(), "kind": "wb_region" if len(members) < 80 else "wb_aggregate", "members": members, "names": {}}
    for il in _wb("incomeLevel"):
        members = [c["id"] for c in _wb(f"country?incomeLevel={il['id']}") if c["id"] in countries]
        if members:
            groups[il["id"]] = {"name": il["value"], "kind": "income", "members": members, "names": {}}
    for code, (name, members) in CURATED.items():
        groups[code] = {"name": name, "kind": "curated", "members": [m for m in members if m in countries], "names": {}}
    print(f"groups: {len(groups)}")

    places = []
    admin1 = {}
    for line in _get("https://download.geonames.org/export/dump/admin1CodesASCII.txt").decode("utf-8").splitlines():
        code, name, ascii_name, _ = line.split("\t")
        iso2 = code.split(".")[0]
        iso3 = next((k for k, v in countries.items() if v["iso2"] == iso2), None)
        if iso3:
            admin1[code] = name
            places.append({"name": name, "ascii": ascii_name, "alt": [], "iso3": iso3, "kind": "admin1", "admin1": name, "population": 0})
    z = zipfile.ZipFile(io.BytesIO(_get("https://download.geonames.org/export/dump/cities15000.zip")))
    for line in z.read("cities15000.txt").decode("utf-8").splitlines():
        f = line.split("\t")
        name, ascii_name, alt, iso2, a1, pop = f[1], f[2], f[3], f[8], f[10], int(f[14] or 0)
        if pop < 150000:
            continue
        iso3 = next((k for k, v in countries.items() if v["iso2"] == iso2), None)
        if iso3:
            alts = [x for x in alt.split(",") if x and len(x) > 2 and _our_script(x)][:300]
            places.append({"name": name, "ascii": ascii_name, "alt": alts, "iso3": iso3, "kind": "city",
                           "admin1": admin1.get(f"{iso2}.{a1}", ""), "population": pop})
    print(f"places: {sum(p['kind'] == 'admin1' for p in places)} admin1, {sum(p['kind'] == 'city' for p in places)} cities")
    GEO.parent.mkdir(parents=True, exist_ok=True)
    GEO.write_text(json.dumps({"countries": countries, "groups": groups, "places": places}, ensure_ascii=False))
    print(f"wrote {GEO} ({GEO.stat().st_size / 1e6:.1f} MB)")


# World Bank aggregates that people search for, with the display name used for them; "World" is a scope, not a group.
KEEP_WB = {
    "Sub-Saharan Africa": "Sub-Saharan Africa", "Middle East, North Africa, Afghanistan & Pakistan": "Middle East, North Africa, Afghanistan & Pakistan",
    "Middle East and North Africa (IFC classification)": "Middle East and North Africa", "South Asia": "South Asia",
    "East Asia & Pacific": "East Asia & Pacific", "Europe & Central Asia": "Europe & Central Asia",
    "Latin America & Caribbean": "Latin America & Caribbean", "North America": "North America", "Low income": "Low income",
    "Lower middle income": "Lower middle income", "Upper middle income": "Upper middle income", "High income": "High income",
    "Low & middle income": "Low & middle income", "Middle income": "Middle income", "Arab World": "Arab World",
    "European Union": "European Union", "Euro area": "Euro area", "OECD members": "OECD members",
    "Least developed countries: UN classification": "Least developed countries", "Small states": "Small states",
    "Pacific island small states": "Pacific island small states", "Caribbean small states": "Caribbean small states",
    "Heavily indebted poor countries (HIPC)": "Heavily indebted poor countries", "Africa": "Africa", "North Africa": "North Africa",
    "Africa Eastern and Southern": "Eastern and Southern Africa", "Africa Western and Central": "Western and Central Africa",
}


def prune() -> None:
    """Keep the groupings people search for and give them plain names."""
    geo = load()
    keep = {}
    for code, g in geo["groups"].items():
        if g["kind"] == "curated":
            keep[code] = g
        elif g["name"] in KEEP_WB:
            keep[code] = {**g, "name": KEEP_WB[g["name"]]}
    geo["groups"] = keep
    GEO.write_text(json.dumps(geo, ensure_ascii=False))
    print(f"groups kept: {len(keep)}")


def load() -> dict:
    return json.loads(GEO.read_text())


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["build", "prune"])
    a = ap.parse_args()
    if a.cmd == "build":
        build()
        prune()
    else:
        prune()


if __name__ == "__main__":
    main()
