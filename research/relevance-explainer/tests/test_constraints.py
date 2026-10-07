"""Render version 3 constraints: thresholds, exclusions and named periods, and their checks."""
import pytest

from relevance_explainer.synth.understand import Understander, check

U = Understander()
POV = {"kind": "indicator", "id": "x", "name": "Poverty headcount ratio at $1.90 a day (2011 PPP) (% of population)",
       "definition": "Share of the population living on less than $1.90 a day at 2011 international prices.",
       "countries": ["Nigeria", "Kenya", "Ghana"], "year_start": 1990, "year_end": 2019, "topics": ["Poverty"]}
STUNT = {"kind": "indicator", "id": "y", "name": "Prevalence of stunting, height for age (% of children under 5)",
         "definition": "Percentage of children under age 5 whose height for age is below the median.",
         "countries": ["Kenya"], "year_start": 2000, "year_end": 2020, "topics": ["Health"]}


def parse(q):
    return U.parse(q, {}, level=3)


@pytest.mark.parametrize("q,value", [("$3 a day poverty Kenya", 3.0), ("people living on less than two dollars a day", 2.0),
                                     ("pauvreté 2,15 $ par jour", 2.15), ("pobreza de 3.65 dólares al día", 3.65)])
def test_money_threshold(q, value):
    t = parse(q)["thresholds"]
    assert any(x["kind"] == "money_per_day" and abs(x["value"] - value) < 1e-6 for x in t)


def test_threshold_checks():
    assert "threshold $3.00 a day: differs (record uses $1.90 a day)" in check(POV, parse("$3 a day poverty Kenya"), U)
    assert "threshold $1.90 a day: matches" in check(POV, parse("poverty at $1.90 a day"), U)


@pytest.mark.parametrize("q,kind,lo,hi", [("stunting in children under 3", "age_under", None, 3),
                                          ("enfants de moins de 5 ans", "age_under", None, 5),
                                          ("youth aged 15-24 unemployment", "age_range", 15, 24),
                                          ("pension coverage 65 and over", "age_over", 65, None)])
def test_ages(q, kind, lo, hi):
    assert {"kind": kind, "lo": lo, "hi": hi} in parse(q)["thresholds"]


def test_age_checks():
    assert "age under 3: differs (record covers age under 5)" in check(STUNT, parse("stunting children under 3"), U)
    assert "age under 5: matches" in check(STUNT, parse("stunting under 5"), U)


@pytest.mark.parametrize("q", ["poverty in West Africa excluding Nigeria", "pauvreté Afrique de l'Ouest sauf le Nigeria",
                               "pobreza excepto Nigeria", "poverty not Nigeria", "Nijerya hariç Batı Afrika yoksulluk"])
def test_excluded_place(q):
    p = parse(q)
    ng = [x for x in p["places"] if x.get("iso3") == "NGA"]
    assert ng and ng[0].get("excluded")
    assert any(c.startswith("excluded place Nigeria") and "includes it" in c for c in check(POV, p, U))


def test_not_excluded_without_cue():
    assert not any(x.get("excluded") for x in parse("poverty in Nigeria and Kenya")["places"])
    assert not any(x.get("excluded") for x in parse("pobreza no Brasil")["places"])  # Portuguese "no" = "in the"


def test_excluded_group():
    p = parse("household income excluding rural areas Kenya")
    assert any(g["value"] == "rural" and g.get("excluded") for g in p["groups"])


@pytest.mark.parametrize("q,start,end", [("unemployment before the 2008 crisis", None, 2007), ("ebola impact Sierra Leone", 2014, 2016),
                                         ("maternal mortality over the past decade", 2016, 2026), ("budget FY2019/20", 2019, 2020)])
def test_named_periods(q, start, end):
    t = parse(q)["time"]
    assert t and t["start"] == start and t["end"] == end


@pytest.mark.parametrize("q,unit", [("HIV incidence rate percentage", "percent"), ("how many people live in Kenya", "count"),
                                    ("porcentaje de cirugías por habitante", "per_capita"), ("dollar value of exports Peru", "currency"),
                                    ("poverty Kenya", None)])
def test_unit_of_query(q, unit):
    assert parse(q).get("unit") == unit


def test_unit_check():
    count = {**POV, "name": "Number of people living in poverty", "unit": "Number"}
    assert "unit a percentage or rate: differs (record gives a count)" in check(count, parse("poverty percentage Kenya"), U)
    assert "unit a percentage or rate: matches" in check(STUNT, parse("stunting percentage"), U)


@pytest.mark.parametrize("q,term", [("business public assistance excluding agricultural firms", "agricultural firms"),
                                    ("import compliance costs excluding agriculture in Kenya", "agriculture"),
                                    ("pobreza en China excepto datos imputados", "datos imputados")])
def test_excluded_terms(q, term):
    assert term in parse(q)["excluded_terms"]


@pytest.mark.parametrize("q", ["kepemilikan tanah di Afrika kecuali wilayah perkotaan", "Canada education except urban areas"])
def test_group_inside_excluded_phrase(q):
    assert any(g["value"] == "urban" and g.get("excluded") for g in parse(q)["groups"])


@pytest.mark.parametrize("q,term", [("无偿家务时间，排除有偿劳动", "有偿劳动"), ("Nijerya hanehalkı anketi hastaneler hariç", "hastaneler"),
                                    ("मालदीव शिक्षा डेटा, स्वास्थ्य को छोड़कर", "स्वास्थ्य")])
def test_excluded_terms_cjk_and_postpositions(q, term):
    assert any(term in t for t in parse(q)["excluded_terms"])


@pytest.mark.parametrize("q", ["jeunesse sans emploi en Ouganda", "численность лиц без гражданства", "jovenes sin estudio ni empleo",
                               "children born without a birth certificate", "Berapa rata-rata kecepatan internet"])
def test_concepts_with_without_are_not_exclusions(q):
    p = parse(q)
    assert p["excluded_terms"] == [] and not any(x.get("excluded") for x in p["places"] + p["groups"])
    assert p.get("unit") is None


def test_excluded_place_not_repeated_as_term():
    assert parse("poverty in West Africa excluding Nigeria")["excluded_terms"] == []


def test_level_2_unchanged():
    p = U.parse("$3 a day poverty excluding Nigeria before the 2008 crisis", {})
    assert "thresholds" not in p and not any(x.get("excluded") for x in p["places"])
