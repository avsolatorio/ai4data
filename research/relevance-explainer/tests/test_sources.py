from relevance_explainer.synth.llm import parse_json
from relevance_explainer.synth.sources import Documents, filter_check, render_input


def _rec(**kw):
    base = dict(kind="indicator", id="X", name="Test series", countries=["Kenya", "Uganda"],
                year_start=1990, year_end=2020, frequency="Annual", year_label="years",
                database="WB_WDI", source="ILO", unit="%", definition="d", concept="", limitation="",
                topics=["Jobs"])
    base.update(kw)
    return base


def test_country_and_year_checks():
    checks = filter_check(_rec(), {"countries": ["Kenya", "Peru"], "years": "2010-2015"})
    assert checks == ["country Kenya: covered", "country Peru: not covered", "years 2010-2015: fully covered"]


def test_country_aliases_match_catalog_spellings():
    for query_name, record_name in [("Egypt", "Egypt, Arab Republic of"), ("Turkey", "Türkiye"),
                                    ("Tanzania", "United Republic of Tanzania"), ("Vietnam", "Viet Nam")]:
        assert filter_check(_rec(countries=[record_name]), {"countries": [query_name]}) == [f"country {query_name}: covered"]


def test_distinct_countries_with_shared_prefix_do_not_match():
    assert filter_check(_rec(countries=["Niger"]), {"countries": ["Nigeria"]}) == ["country Nigeria: not covered"]


def test_regional_document_is_not_reported_as_uncovered():
    doc = _rec(kind="document", countries=["Africa"])
    assert "covers a region or the world" in filter_check(doc, {"countries": ["Kenya"]})[0]


def test_truncated_multi_country_list_cannot_refute_coverage():
    study = _rec(kind="study", countries=["Afghanistan", "Albania"], countries_truncated=True)
    assert filter_check(study, {"countries": ["Albania"]}) == ["country Albania: covered"]
    assert "cannot confirm" in filter_check(study, {"countries": ["Indonesia"]})[0]


def test_messy_document_country_field_still_matches():
    doc = _rec(kind="document", countries=["Arab Republic of,Egypt, Arab Republic of"])
    assert filter_check(doc, {"countries": ["Egypt"]}) == ["country Egypt: covered"]


def test_country_named_in_description_is_not_called_uncovered():
    study = _rec(kind="study", countries=["Kyrgyz Republic"], abstract="Polls in Kazakhstan, Kyrgyzstan and Tajikistan.")
    assert "named in the description" in filter_check(study, {"countries": ["Tajikistan"]})[0]


def test_partly_covered_range():
    assert filter_check(_rec(), {"years": "2015-2025"}) == ["years 2015-2025: partly covered"]


def test_point_record_inside_and_outside_range():
    study = _rec(kind="study", year_start=2019, year_end=2019, year_label="survey years")
    assert filter_check(study, {"years": "2010-2020"}) == ["survey years 2010-2020: within the range"]
    assert filter_check(study, {"years": "2021-2023"}) == ["survey years 2021-2023: outside the range"]


def test_document_years_are_not_checked_against_publication_year():
    doc = _rec(kind="document", year_start=2019, year_end=2019, year_label="publication year")
    assert filter_check(doc, {"years": "2010-2020"}) == ["years 2010-2020: period studied not verifiable from metadata (published 2019)"]


def test_frequency_mismatch_names_actual_frequency():
    assert filter_check(_rec(), {"frequency": "Quarterly"}) == ["frequency Quarterly: differs (Annual)"]


def test_missing_coverage_is_not_reported_as_uncovered():
    assert filter_check(_rec(countries=[]), {"countries": ["Kenya"]}) == ["country Kenya: coverage not stated"]


def test_render_input_includes_language_and_checks():
    text = render_input("emploi des jeunes", {"years": "2019"}, _rec(), "French")
    assert text.splitlines()[0] == "Respond in: French"
    assert "Filter check: years 2019: fully covered" in text


def test_passage_skips_front_matter_and_needs_term_overlap():
    body = " ".join(["filler"] * 800) + " maternal employment rises with preschool access in Indonesia. " + " ".join(["words"] * 300)
    text = " ".join(["cover"] * 800) + " " + body
    assert "preschool" in Documents.passage(text, "preschool access and maternal employment")
    assert Documents.passage(text, "volcano eruption monitoring") is None


def test_similarity_line_goes_after_filters():
    from relevance_explainer.synth.signals import add_similarity_line

    text = "Respond in: French\nItem type: x\nQuery: q\nFilters: none\nResult: r"
    out = add_similarity_line(text, 0.6149).split("\n")
    assert out[out.index("Filters: none") + 1] == "Retrieval similarity: 0.61"
    assert len(out) == 6


def test_parse_json_tolerates_fences_and_prose():
    assert parse_json('```json\n{"a": 1}\n```') == {"a": 1}
    assert parse_json('Here you go: [{"a": 1}] thanks') == [{"a": 1}]
