from relevance_explainer.synth.tldr_grade import check

ROW = {"input": ("Item type: indicator series (Data360)\nResult: Unemployment rate\nYears: 2015-2023\nCoverage: Kenya\n"
                 "Respond in: English\nSearch: unemployment Kenya 2018\nFacets: none\n"
                 "Understood: place=Kenya (country) [query]; time=2018 [query]\nChecks: place Kenya: covered"),
       "output": ("relevant\nThe series reports the unemployment rate for Kenya from 2015 to 2023, which includes the year "
                  "2018 that the search asks for, so it answers the search directly."),
       "language": "English"}


def test_good_answer_passes():
    assert check({"tldr": "Kenya's unemployment rate is reported for 2015 to 2023, including 2018.", "grade": 9}, ROW) == []


def test_grade_outside_the_verdict_range_fails():
    assert any("outside the range" in b for b in check({"tldr": "Kenya's unemployment for 2018 is covered.", "grade": 2}, ROW))


def test_invented_number_and_place_fail():
    bad = check({"tldr": "Covers Kenya and Tanzania from 1990.", "grade": 8}, ROW)
    assert any("does not" in b for b in bad)


def test_two_sentences_and_wrong_language_fail():
    assert any("one sentence" in b for b in check({"tldr": "It covers Kenya. It includes 2018.", "grade": 8}, ROW))
    assert any("not in English" in b for b in check(
        {"tldr": "La serie cubre la tasa de desempleo de Kenia entre 2015 y 2023.", "grade": 8}, ROW))
