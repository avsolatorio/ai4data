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


def test_claim_source_names_the_field_that_holds_the_value():
    from relevance_explainer.answer_c import _checked
    from relevance_explainer.claims import record_fields
    c = {x["value"]: x["source"] for x in _checked(verify(from_model(OUT), INPUT), record_fields(INPUT))["claims"]}
    assert c == {"South Africa": "Coverage", "2000-2023": "Years"}


def test_parse_checks_states():
    from relevance_explainer.answer_c import parse_checks
    text = ("Result: x\nChecks: place Kenya: covered | time 2010: not covered (item year 2020) | "
            "type microdata: item is an indicator series | place India: India covered; subnational detail not stated")
    assert [(c["constraint"], c["state"]) for c in parse_checks(text)] == [
        ("place Kenya", "met"), ("time 2010", "unmet"), ("type microdata", "info"), ("place India", "info")]
    assert parse_checks("Result: x\nChecks: none") == []


def _o(verified=True, band="high", vset=("relevant",), verdict="relevant", errors=()):
    return {"verdict": verdict, "verdict_set": list(vset), "faith": {"score": 0.9, "band": band},
            "check": {"all_verified": verified, "errors": list(errors)}}


def test_trust_states():
    from relevance_explainer.answer_c import trust
    assert trust(_o())["state"] == "checked"
    assert trust(_o(band="check"))["state"] == "check"
    assert trust(_o(verified=False))["state"] == "check"
    assert trust(_o(band="low"))["state"] == "check"
    assert trust(_o(band="low", verified=False))["state"] == "withheld"
    assert trust(_o(verdict="invalid"))["state"] == "withheld"
    assert trust(_o(vset=("relevant", "partly_relevant")))["state"] == "checked"  # verdict doubt is shown on the verdict


def test_verdict_view_hedges_two_verdict_sets():
    from relevance_explainer.answer_c import verdict_view
    assert verdict_view(_o())["label"] == "Matches the search"
    v = verdict_view(_o(verdict="irrelevant", vset=("partly_relevant", "irrelevant")))
    assert v["uncertain"] and v["label"] == "May partly match" and v["between"] == ["partly_relevant", "irrelevant"]
    assert verdict_view(_o(vset=("relevant", "partly_relevant", "irrelevant")))["label"] == "Unclear"


def test_explain_page_and_feedback(tmp_path, monkeypatch):
    import json

    from starlette.testclient import TestClient

    from relevance_explainer.webapp import server
    monkeypatch.setattr(server, "FEEDBACK_LOG", tmp_path / "fb.jsonl")
    c = TestClient(server.app)
    page = c.get("/explain")
    assert page.status_code == 200 and "Search result explanations" in page.text
    assert c.post("/api/feedback", json={"helpful": "yes"}).status_code == 400
    assert c.post("/api/feedback", json={"helpful": False, "feedback_id": "a1", "comment": "x" * 3000}).json() == {"ok": True}
    row = json.loads((tmp_path / "fb.jsonl").read_text())
    assert row["helpful"] is False and row["feedback_id"] == "a1" and len(row["comment"]) == 2000
