from relevance_explainer.tiny.faithfulness import rate, unsupported

SRC = "Query: maternal deaths Brazil 2024\nResult: Maternal mortality ratio\nYears: 1960-2023\nCoverage: 185 countries"


def test_supported_numbers_and_names_pass():
    assert unsupported("The series covers 1960-2023 and 185 countries, including Brazil.", SRC) == []


def test_invented_number_and_name_are_flagged():
    out = unsupported("It reports 1996 data for Uganda over 207 countries.", SRC)
    assert "1996" in out and "207" in out and "Uganda" in out


def test_accents_and_case_are_ignored_when_matching():
    assert unsupported("El indicador cubre Brasil.", "Query: brasil") == []


def test_sentence_initial_capitals_are_not_names():
    assert unsupported("Maternal data are annual. However, coverage is partial.", SRC) == []


def test_rate_counts_narratives():
    r = rate([("Brazil 1960", SRC), ("Uganda 1996", SRC)])
    assert r["n"] == 2 and r["narratives_with_unsupported"] == 0.5
