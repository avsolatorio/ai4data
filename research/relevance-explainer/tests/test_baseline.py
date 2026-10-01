from relevance_explainer import Correspondence, MatchDiagnostics, MatchLevel, explain


def _example(**kw):
    base = dict(
        query="employment of young women in rural areas",
        result_title="Labor Force Participation Rate, Female, Ages 15-24",
        similarity=0.81,
        matched=[
            Correspondence("employment", "labor force participation", "indicator", MatchLevel.SYNONYM),
            Correspondence("young", "ages 15-24", "age", MatchLevel.SYNONYM),
            Correspondence("women", "female", "sex", MatchLevel.SYNONYM),
        ],
        missing=["rural"],
    )
    base.update(kw)
    return MatchDiagnostics(**base)


def test_partial_match_names_missing_concept():
    text = explain(_example())
    assert "partial match" in text
    assert "rural" in text


def test_strong_match_when_nothing_missing():
    assert "strong match" in explain(_example(missing=[]))


def test_low_similarity_with_overlap_is_weak():
    assert "weak match" in explain(_example(missing=[], similarity=0.3))


def test_no_matches_reports_weak_similarity_only():
    text = explain(_example(matched=[], missing=["rural"]))
    assert "no matched concepts" in text


def test_to_dict_serializes_level_values():
    assert _example().to_dict()["matched"][0]["level"] == "synonym"
