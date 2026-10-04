import pytest

from relevance_explainer.synth.understand import Understander, check, norm


@pytest.fixture(scope="module")
def u():
    return Understander()


def places(u, q):
    return [(p["kind"], p.get("iso3") or p.get("code")) for p in u.parse(q)["places"]]


@pytest.mark.parametrize("q,expected", [
    ("pobreza rural en Cochabamba despues de 2015", [("admin1", "BOL")]),
    ("latest unemployment MENA", [("group", "CME")]),
    ("girls school enrollment sahel 2010-2020", [("group", "SAHEL_G5")]),
    ("معدل البطالة في المغرب منذ 2015", [("country", "MAR")]),
    ("肯尼亚 农村 贫困 数据", [("country", "KEN")]),
    ("भारत में ग्रामीण महिलाओं की साक्षरता दर", [("country", "IND")]),
    ("inflation monthly Turkey", [("country", "TUR")]),
    ("Kenya'da kadın istihdamı", [("country", "KEN")]),
    ("инфляция в Казахстане", [("country", "KAZ")]),
    ("здравоохранение в Нигерии и Гане", [("country", "NGA"), ("country", "GHA")]),
    ("данные по Узбекистану за 2019 год", [("country", "UZB")]),
    ("refugees in Jordan and Lebanon last 5 years", [("country", "JOR"), ("country", "LBN")]),
    ("kemiskinan di Papua tahun 2018", [("admin1", "IDN")]),
    ("GDP per capita", []),
])
def test_places(u, q, expected):
    assert sorted(places(u, q)) == sorted(expected)


def test_time_and_groups(u):
    p = u.parse("girls school enrollment sahel 2010-2020")
    assert (p["time"]["type"], p["time"]["start"], p["time"]["end"]) == ("range", 2010, 2020)
    assert [g["value"] for g in p["groups"]] == ["female"]
    assert u.parse("données démographiques Côte d'Ivoire années 90")["time"]["start"] == 1990
    assert u.parse("uchumi wa kenya tangu 2010")["time"]["type"] == "since"
    assert u.parse("latest unemployment MENA")["time"]["type"] == "latest"
    assert u.parse("pre-covid tourism arrivals caribbean small states")["time"]["end"] == 2019


def test_facets_merge(u):
    p = u.parse("poverty", {"countries": ["Kenya"], "years": "2015"})
    assert p["places"][0]["iso3"] == "KEN" and p["places"][0]["source"] == "facet"
    assert p["time"]["source"] == "facet" and p["time"]["start"] == 2015


def test_check_statuses(u):
    rec = {"kind": "indicator", "name": "Poverty headcount, rural", "countries": ["Kenya", "Uganda"], "year_start": 2000,
           "year_end": 2018, "frequency": "Annual", "definition": "Share of the rural population below the line."}
    lines = check(rec, u.parse("rural poverty kenya tanzania since 2015 annual"), u)
    txt = " | ".join(lines)
    assert "place Kenya: covered" in txt and "place Tanzania: not covered" in txt
    assert "time 2015 or later: partly covered (2000-2018)" in txt
    assert "frequency annual: matches" in txt and "group rural: mentioned in the metadata" in txt


def test_norm_keeps_devanagari_marks():
    assert norm("Côte") == "cote" and norm("भारत में") == "भारत में"
