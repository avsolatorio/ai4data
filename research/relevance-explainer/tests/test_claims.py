from relevance_explainer.claims import annotate, parse_record, record_fields, strip_refs, verify

INPUT = ("Item type: indicator series (Data360)\nResult: Unemployment, total (% of labor force)\n"
         "Database: World Development Indicators; Source: World Bank\nYears: 2015-2023; Frequency: annual\n"
         "Coverage: Kenya, Uganda\nRespond in: Spanish\nSearch: desempleo en Kenia 2018\nFacets: none\n"
         "Understood: place=Kenya (country) [query]; time=2018 [query]\n"
         "Checks: place Kenya: covered | time 2018: covered (2015-2023)")


def test_parse_record_splits_pairs_and_skips_search_fields():
    pairs = dict(parse_record(INPUT))
    assert pairs["Database"] == "World Development Indicators" and pairs["Source"] == "World Bank"
    f = record_fields(INPUT)
    assert "Search" not in f and "Understood" not in f and f["Years"] == "2015-2023"


def test_annotate_round_trip_and_verify():
    text = "La serie cubre Kenia entre 2015 y 2023, e incluye el año 2018."
    s = annotate("relevant", text, INPUT)
    ck = verify(s, INPUT)
    assert ck.sections["exp"] == text and "<r" not in strip_refs(s)
    assert ck.refs and ck.all_verified, (s, ck)


def test_wrong_reference_is_not_verified():
    s = ("<verdict>relevant</verdict>\n<claims><c1>Coverage: Kenya</c1></claims>\n"
         "<exp>Cubre <r1>Tanzania</r1>.</exp>")
    ck = verify(s, INPUT)
    assert ck.claims[1].ok and not ck.refs[0].ok


def test_claim_absent_from_its_field_fails_closed():
    s = ("<verdict>relevant</verdict>\n<claims><c1>Years: 1990-2000</c1><c2>Coverage: 2015</c2></claims>\n"
         "<exp>De <r1>1990 a 2000</r1>; <r3>Kenia</r3>.</exp>")
    ck = verify(s, INPUT)
    assert not ck.claims[1].ok and not ck.claims[2].ok
    assert [r.ok for r in ck.refs] == [False, False] and ck.refs[1].reason == "no claim c3"


def test_numbers_and_places_outside_references_are_reported():
    s = ("<verdict>relevant</verdict>\n<claims><c1>Coverage: Kenya</c1></claims>\n"
         "<exp>Cubre <r1>Kenia</r1> desde 1980 y también Tanzania.</exp>")
    ck = verify(s, INPUT)
    assert ck.refs[0].ok and "1980" in ck.unverified and any("tanzania" in u.lower() for u in ck.unverified)


def test_year_range_in_another_language_matches_claim():
    s = ("<verdict>relevant</verdict>\n<claims><c1>Years: 2015-2023</c1></claims>\n"
         "<exp>Datos de <r1>2015 a 2023</r1>.</exp>")
    assert verify(s, INPUT).all_verified


def test_consistency_flags():
    bad_grade = "<verdict grade=\"1\">relevant</verdict>\n<claims></claims>\n<exp>Sí.</exp>"
    assert any("grade" in f for f in verify(bad_grade, INPUT).flags)
    failed = INPUT.replace("place Kenya: covered", "place Kenya: not covered")
    ok = "<verdict>relevant</verdict>\n<claims></claims>\n<exp>Sí.</exp>"
    assert any("check failed" in f for f in verify(ok, failed).flags)
