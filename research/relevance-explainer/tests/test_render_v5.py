"""Render version 5 (notebook 2026-10-16): place filters and plural subnational words, the language guard, the opendata
item type and its CKAN adapter, and the unseen-catalog test set. Version 4, the training and evaluation format, must
not change."""
import csv
import json
from pathlib import Path

import pytest

from relevance_explainer.synth.understand import Understander, check

ROOT = Path(__file__).resolve().parents[1]
U = Understander()


def places(q, level):
    return [p["name"] for p in U.parse(q, {}, level=level)["places"]]


@pytest.mark.parametrize("q,false,kept", [
    ("Indigenous languages spoken in Nunavut", "Spokane", "Nunavut"),
    ("Türkiye okuma yazma oranı", "Oran", "Turkiye"),
    ("kahirapan sa Pilipinas", "South Africa", "Philippines"),
    ("saúde dos recifes de coral na Austrália 2023", "North America", "Australia"),
    ("langues autochtones parlées à la maison Québec", "Arrah", "Quebec"),
    ("takwimu za umaskini Kenya", "South Africa", "Kenya"),
])
def test_level5_drops_false_places_and_level4_keeps_them(q, false, kept):
    assert false in places(q, 4) and kept in places(q, 4)
    assert false not in places(q, 5) and kept in places(q, 5)


@pytest.mark.parametrize("q,place", [
    ("SA unemployment", "South Africa"), ("CAN trade agreements", "Andean countries"), ("malaria nampula", "Nampula"),
    ("Москва бедность", "Moscow"), ("unemployment uk", "United Kingdom"), ("pobreza en ciudad del cabo", "Cape Town"),
    ("contaminación en belgrado", "Belgrade"),
])
def test_level5_keeps_real_places(q, place):
    assert place in places(q, 5)


def test_level5_counts_plural_subnational_words():
    rec = {"kind": "opendata", "name": "Housing starts, all areas, Canada and provinces", "countries": ["Canada"],
           "notes": "Monthly housing starts."}
    st = {lv: next(c for c in check(rec, U.parse("housing starts Ontario", {}, level=lv), U) if c.startswith("place Ontario"))
          for lv in (4, 5)}
    assert st[4].endswith("subnational detail not stated")
    assert st[5].endswith("subnational breakdown mentioned, this place not named")


def test_render_defaults_to_version_5_and_keeps_version_4():
    from relevance_explainer.inference import RENDER_VERSION, render
    rec = {"kind": "indicator", "name": "Poverty headcount ratio", "database": "WDI", "source": "World Bank", "unit": "%",
           "frequency": "Annual", "definition": "Share of the population below the poverty line.", "concept": None,
           "limitation": None, "topics": [], "year_start": 2000, "year_end": 2022, "countries": ["Kenya"], "year_label": "years"}
    v5, v4 = render(rec, "takwimu za umaskini Kenya", "Swahili"), render(rec, "takwimu za umaskini Kenya", "Swahili", render_version=4)
    assert RENDER_VERSION == 5 and "South Africa" in v4 and "South Africa" not in v5
    assert [l for l in v4.split("\n") if not l.startswith(("Understood:", "Checks:"))] == \
        [l for l in v5.split("\n") if not l.startswith(("Understood:", "Checks:"))]


LID = (ROOT / "data/geo/lid.176.ftz").exists()


@pytest.mark.parametrize("lang,q,refused,needs_lid", [
    ("German", "Grundwasser Europa", True, False), ("English", "日本の平均寿命", True, False),
    ("English", "한국 실업률 통계", True, False), ("English", "tỷ lệ nghèo đói ở Việt Nam", True, True),
    ("English", "Arbeitslosigkeit in Deutschland nach Bundesland", True, True),
    ("Chinese", "中国 城镇化率", False, False), ("English", "malaria nampula", False, False),
    ("Swahili", "takwimu za umaskini wa kaya Kenya 2019", False, False), ("Hindi", "भारत में मातृ मृत्यु दर", False, False),
    ("Turkish", "Türkiye okuma yazma oranı", False, False), ("Indonesian", "sampah plastik per kapita Indonesia 2020", False, False),
])
def test_language_guard(lang, q, refused, needs_lid):
    from relevance_explainer.language import unsupported
    if needs_lid and not LID:
        pytest.skip("data/geo/lid.176.ftz is missing (scripts/fetch_resources.sh)")
    assert (unsupported(f"Item type: x\nRespond in: {lang}\nSearch: {q}") is not None) == refused


def test_unsupported_language_withholds_the_explanation():
    from relevance_explainer.answer_c import trust
    o = {"verdict": "relevant", "verdict_set": ["relevant"], "faith": {"score": 0.9, "band": "high"},
         "check": {"all_verified": True, "errors": []}, "unsupported_language": "The model does not write explanations in German."}
    assert trust(o) == {"state": "withheld", "reasons": ["The model does not write explanations in German."]}


def test_ckan_record_and_opendata_renderer():
    from relevance_explainer.schemas import RENDERERS, ckan_record
    pkg = {"name": "x1", "title_translated": {"en": "Housing starts", "fr": "Mises en chantier"},
           "notes": "<p>Monthly housing starts by province.</p>", "organization": {"title": "CMHC | SCHL"},
           "keywords": {"en": ["housing"]}, "frequency": "P1M", "time_period_coverage_start": "1990-01-01",
           "resources": [{"format": "CSV"}]}
    r = ckan_record(pkg, ["Canada"])
    assert (r["kind"], r["name"], r["organization"], r["frequency"], r["year_start"], r["year_end"]) == \
        ("opendata", "Housing starts", "CMHC", "monthly", 1990, None)
    assert "Years: from 1990" in "\n".join(RENDERERS["opendata"].render(r)) and RENDERERS["opendata"].label.startswith("dataset")
    harvested = ckan_record({"title": "Solar 2017", "temporal_coverage_from": "2025-03-06 09:24:59.454216"}, ["Australia"])
    epoch = ckan_record({"title": "Drought", "temporal_coverage_from": "1970-01-01 00:00:00"}, ["Australia"])
    assert harvested["year_start"] is None and epoch["year_start"] is None
    chk = check(r, U.parse("housing dataset Canada", {}, level=5), U)
    assert not any(c.startswith("type") and "item is" in c for c in chk)


def test_unseen_catalog_set_is_complete():
    d = ROOT / "evalsets" / "unseen_catalogs"
    recs = {json.loads(l)["id"]: json.loads(l) for l in open(d / "records.jsonl", encoding="utf-8")}
    rows = list(csv.DictReader(open(d / "cases.tsv", encoding="utf-8"), delimiter="\t"))
    verdicts = {"relevant", "partly_relevant", "irrelevant"}
    assert len(rows) == 49 and {r["record"] for r in rows} == set(recs)
    assert all(r["label"] in verdicts and r["also_ok"] in verdicts | {""} and r["group"] in ("supported", "unsupported") for r in rows)
    from relevance_explainer.inference import render
    for r in rows:
        assert f"Search: {r['search']}" in render(recs[r["record"]], r["search"], r["language"], understander=U)
