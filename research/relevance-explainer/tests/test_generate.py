import random

from relevance_explainer.synth.generate import _clean, _informative, _items, _language, _safe, balance, split_of


def test_language_codes_are_canonicalized_and_unknown_dropped():
    assert _language("ar") == "Arabic"
    assert _language("en") == "English"
    assert _language("french") == "French"
    assert _language("Klingon") is None


def test_clean_drops_rows_with_unknown_language_or_bad_filters():
    assert _clean({"query": "q", "language": "Klingon"}) is None
    assert _clean({"query": "q", "filters": ["not", "a", "dict"]}) is None
    assert _clean({"query": "q", "language": "fr"})["language"] == "French"


def test_items_unwraps_dicts_and_skips_non_objects():
    assert _items('[{"query": "a"}, "stray string", 3]') == [{"query": "a"}]
    assert _items('{"searches": [{"query": "a"}]}') == [{"query": "a"}]
    assert _items('{"query": "a"}') == [{"query": "a"}]


def test_safe_returns_empty_on_failure():
    def boom(_):
        raise ValueError("bad")

    assert _safe(boom, ("seed", "study")) == []


def test_split_is_deterministic_per_theme():
    assert split_of("GDP growth") == split_of("GDP growth")
    assert {split_of(f"theme {i}") for i in range(200)} == {"train", "val", "test"}


def test_balance_caps_the_largest_class():
    rows = [{"verdict": "irrelevant"}] * 100 + [{"verdict": "partly_relevant"}] * 20 + [{"verdict": "relevant"}] * 10
    out = balance(rows, 1.5, random.Random(0))
    counts = {v: sum(r["verdict"] == v for r in out) for v in ("irrelevant", "partly_relevant", "relevant")}
    assert counts == {"irrelevant": 15, "partly_relevant": 15, "relevant": 10}


def test_sparse_records_are_mostly_dropped_and_rich_ones_kept():
    rng = random.Random(0)
    sparse = {"abstract": "", "topics": []}
    rich = {"abstract": "A long enough abstract describing the survey content in detail.", "topics": []}
    assert all(_informative(rich, rng) for _ in range(50))
    kept = sum(_informative(sparse, rng) for _ in range(400))
    assert 20 < kept < 100  # about 15%
