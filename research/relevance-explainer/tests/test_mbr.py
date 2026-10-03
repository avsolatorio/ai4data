from relevance_explainer.tiny.mbr import chrf, select


def test_chrf_bounds():
    assert abs(chrf("the same text", "the same text") - 1.0) < 1e-9
    assert chrf("abc", "xyz") == 0.0
    assert 0 < chrf("covers Kenya 2015", "covers Kenya 2019") < 1


def test_select_picks_the_consensus_candidate():
    cands = ["The survey covers Kenya in 2015.", "The survey covers Kenya in 2015 only.", "Zebra quantum blue.",
             "The survey covers Kenya, 2015."]
    assert select(cands) in (0, 1, 3)
    assert select(["only"]) == 0
