import re

from relevance_explainer.claims import (CATEGORIES, SPECIAL_TOKENS, annotate, from_model, parse, parse_record, record_fields,
                                        plain_output, sanitize_input, strip_model_tags, strip_refs, to_model, verify)

INPUT = ("Item type: indicator series (Data360)\nResult: Unemployment, total (% of labor force)\n"
         "Database: World Development Indicators; Source: World Bank\nYears: 2015-2023; Frequency: annual\n"
         "Coverage: Kenya, Uganda\nRespond in: Spanish\nSearch: desempleo en Kenia 2018\nFacets: none\n"
         "Understood: place=Kenya (country) [query]; time=2018 [query]\n"
         "Checks: place Kenya: covered | time 2018: covered (2015-2023)")


def c(cid, field, value):
    return f'<c cid="{cid}" field="{field}">{value}</c>'


def out(claims, exp, verdict="relevant", grade=None):
    g = f' grade="{grade}"' if grade is not None else ""
    return f"<verdict{g}>{verdict}</verdict>\n<claims>{''.join(claims)}</claims>\n<exp>{exp}</exp>"


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
    assert all(cl.field in CATEGORIES for cl in ck.claims.values())


def test_ids_are_random_but_reproducible():
    text = "Cubre Kenia y Uganda de 2015 a 2023."
    assert annotate("relevant", text, INPUT) == annotate("relevant", text, INPUT)
    seen = {tuple(sorted(parse(annotate("relevant", text, INPUT, seed=s)).claims)) for s in range(20)}
    assert len(seen) > 5  # different seeds give different ids
    for s in range(5):
        assert verify(annotate("relevant", text, INPUT, seed=s), INPUT).all_verified


def test_wrong_reference_is_not_verified():
    ck = verify(out([c(7, "geography", "Kenya")], 'Cubre <r cid="7">Tanzania</r>.'), INPUT)
    assert ck.claims[7].ok and not ck.refs[0].ok


def test_claim_absent_from_its_category_fails_closed():
    ck = verify(out([c(4, "time", "1990-2000"), c(9, "geography", "2015")],
                    'De <r cid="4">1990 a 2000</r>; <r cid="11">Kenia</r>.'), INPUT)
    assert not ck.claims[4].ok and not ck.claims[9].ok
    assert [r.ok for r in ck.refs] == [False, False] and ck.refs[1].reason == "no claim 11"


def test_category_accepts_any_field_of_its_kind():
    assert verify(out([c(2, "source", "World Bank")], 'Del <r cid="2">World Bank</r>.'), INPUT).all_verified


def test_numbers_and_places_outside_references_are_reported():
    ck = verify(out([c(1, "geography", "Kenya")], 'Cubre <r cid="1">Kenia</r> desde 1980 y también Tanzania.'), INPUT)
    assert ck.refs[0].ok and "1980" in ck.unverified and any("tanzania" in u.lower() for u in ck.unverified)


def test_year_range_in_another_language_matches_claim():
    assert verify(out([c(5, "time", "2015-2023")], 'Datos de <r cid="5">2015 a 2023</r>.'), INPUT).all_verified


def test_consistency_flags():
    assert any("grade" in f for f in verify(out([], "Sí.", grade=1), INPUT).flags)
    failed = INPUT.replace("place Kenya: covered", "place Kenya: not covered")
    assert any("check failed" in f for f in verify(out([], "Sí."), failed).flags)


def test_model_form_round_trip():
    s = annotate("relevant", "Cubre Kenia de 2015 a 2023.", INPUT, grade=8, tldr="Kenia, 2015 a 2023.")
    m = to_model(s)
    assert m.split("\n")[0] == "relevant" and m.split("\n")[1] == "<|g:8|>"  # the verdict stays the first line
    assert 'cid="' not in m and "<|cid:" in m and from_model(m) == s
    assert verify(from_model(m), INPUT).all_verified


def test_special_tokens_are_unique_and_inputs_cannot_open_tags():
    assert len(SPECIAL_TOKENS) == len(set(SPECIAL_TOKENS))
    assert "<|" not in sanitize_input("a <|c|> b") and "1" not in strip_model_tags("<|cid:1|>x<|g:9|>").replace("x", "")


def test_words_after_tags_keep_their_leading_space():
    s = annotate("relevant", "Cubre Kenia de 2015 a 2023.", INPUT, grade=8)
    m = to_model(s)
    assert " <|r|>" not in m and "|> Kenia" in m  # the space before a reference sits inside the tag
    assert all(v.startswith(" ") for v in re.findall(r"<\|f:[a-z]+\|>([^<]*)", m))
    assert from_model(m) == s and plain_output(m) == "relevant\nCubre Kenia de 2015 a 2023."
    old = m.replace("Cubre<|r|>", "Cubre <|r|>").replace("|> Kenia", "|>Kenia")  # the earlier form still reads
    assert verify(from_model(old), INPUT).all_verified


def test_values_in_scripts_without_spaces_get_none():
    m = to_model('<verdict>relevant</verdict>\n<claims><c cid="2" field="geography">肯尼亚</c></claims>\n<exp>覆盖<r cid="2">肯尼亚</r>。</exp>')
    assert "<|f:geography|>肯尼亚" in m and "覆盖<|r|><|cid:2|>肯尼亚" in m


def test_number_claim_takes_the_closest_written_form():
    inp = INPUT.replace("Coverage: Kenya, Uganda", "Coverage: Kenya, Uganda; Retrieved passage: 29 roads, density 2.9 km")
    s = annotate("relevant", "La densidad es 2,9 km.", inp)
    assert 'field="text">2.9</c>' in s and verify(s, inp).all_verified  # not "29", whatever the set order
