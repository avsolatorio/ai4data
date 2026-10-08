"""Tests for the scripts shipped with the serving-statistics-to-agents cookbook."""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
SCRIPTS = REPO / "website" / "static" / "cookbook-files" / "serving-statistics-to-agents"


def load_script(name: str):
    spec = importlib.util.spec_from_file_location(name, SCRIPTS / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def manifest_check():
    return load_script("check_tool_manifest")


@pytest.fixture(scope="module")
def agent_eval():
    return load_script("agent_eval")


def test_example_manifest_passes(manifest_check, capsys):
    assert manifest_check.main([str(SCRIPTS / "tool_manifest.json")]) == 0
    out = capsys.readouterr().out
    assert "statistics-example 1.0.0: 4 tool(s), 3 resource(s)" in out
    assert "0 error(s), 0 warning(s)" in out


def test_manifest_rules_catch_design_faults(manifest_check, tmp_path, capsys):
    m = json.loads((SCRIPTS / "tool_manifest.json").read_text(encoding="utf-8"))
    m["tools"][0]["name"] = "SearchSeries"
    m["tools"][0]["description"] = "Search."
    m["tools"][2]["readOnlyHint"] = False
    m["tools"][2]["outputs"].remove("SOURCE_URL")
    del m["tools"][3]["example"]
    m["resources"] = []
    bad = tmp_path / "bad.json"
    bad.write_text(json.dumps(m), encoding="utf-8")
    assert manifest_check.main([str(bad)]) == 1
    out = capsys.readouterr().out
    for expected in (
        "SearchSeries: name must be a snake_case verb phrase",
        "description has 1 words",
        "get_observations: not marked read-only",
        "lacks provenance fields ['SOURCE_URL']",
        "list_code_values: no example call",
        "no guidance resource",
    ):
        assert expected in out


def test_agent_eval_scores_the_example_traces(agent_eval, capsys):
    assert agent_eval.main([str(SCRIPTS / "agent_questions.csv"), str(SCRIPTS / "agent_traces.jsonl")]) == 1
    out = capsys.readouterr().out
    assert "tools   7/8" in out
    assert "value   4/5" in out
    assert "behave  7/8" in out
    assert "a05: value failed" in out  # the French answer gives 6.3 for a quarter whose value is 6.1
    assert "a07: tools failed" in out and "a07: behave failed" in out  # answered a regional figure the series does not have


def test_number_parsing_handles_french_decimals_and_thousands(agent_eval):
    assert agent_eval.numbers_in("6,3 % et 12,500 ménages en 2025") == [6.3, 12500.0, 2025.0]
    assert agent_eval.numbers_in("7.6% in 2024") == [7.6, 2024.0]
