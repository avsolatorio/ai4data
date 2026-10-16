from relevance_explainer.answer_c import _band, segments
from relevance_explainer.claims import from_model, verify

INPUT = ("Item type: statistical indicator series\nResult: Vitamin A supplementation coverage rate\n"
         "Coverage: South Africa; Years: 2000-2023\nSearch: vitamin A in South Africa")
OUT = ("partly_relevant\n<|g:6|>\n<|claims|><|c|><|cid:4|><|f:geography|>South Africa<|/c|>"
       "<|c|><|cid:9|><|f:time|>2000-2023<|/c|><|/claims|>\n<|tldr|>Covers <|r|><|cid:4|>South Africa<|/r|>.<|/tldr|>\n"
       "<|exp|>Data for <|r|><|cid:4|>South Africa<|/r|> from <|r|><|cid:9|>2000-2023<|/r|> and <|r|><|cid:7|>Kenya<|/r|>.<|/exp|>")


def test_segments_follow_the_references_in_order():
    ck = verify(from_model(OUT), INPUT)
    refs = [{"ok": r.ok, "reason": r.reason} for r in ck.refs if r.section == "exp"]
    raw = from_model(OUT).split("<exp>")[1].split("</exp>")[0]
    segs = segments(raw, refs)
    assert "".join(s["text"] for s in segs) == "Data for South Africa from 2000-2023 and Kenya."
    marked = [(s["text"], s["claim"], s["ok"]) for s in segs if "claim" in s]
    assert marked == [("South Africa", 4, True), ("2000-2023", 9, True), ("Kenya", 7, False)]


def test_faith_bands():
    assert [_band(x) for x in (0.95, 0.8, 0.79, 0.5, 0.49, 0.0)] == ["high", "high", "check", "check", "low", "low"]
