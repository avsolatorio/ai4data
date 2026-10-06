"""Item-line cleaning versions: version 1 is kept for rows narrated against it; version 2 drops only empty fields."""
from relevance_explainer.synth.realistic import RENDER_VERSION, _clean_item_lines, renderer

LINES = ["Result: X", "Database: n/a; Source: WDI", "Unit: n/a; Frequency: Annual; Years: 1960-2020",
         "Coverage: 1 country: Maldives; Geographic scope: n/a", "Abstract: n/a", "Concept: a; b: c"]


def test_version_1_is_unchanged():
    assert _clean_item_lines(LINES) == ["Result: X", "Database: n/a; Source: WDI",
                                        "Unit: n/a; Frequency: Annual; Years: 1960-2020", "Concept: a; b: c"]


def test_version_2_keeps_values_and_drops_empty_fields():
    assert _clean_item_lines(LINES, 2) == ["Result: X", "Source: WDI", "Frequency: Annual; Years: 1960-2020",
                                           "Coverage: 1 country: Maldives", "Concept: a; b: c"]
    assert RENDER_VERSION == 2


def test_renderer_covers_trained_and_schema_kinds():
    assert renderer("study").label == "survey microdata study"
    assert renderer("table").label == "statistical table"
