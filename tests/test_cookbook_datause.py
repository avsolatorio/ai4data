"""Tests for the scripts shipped with the monitoring-data-use cookbook.

The scripts live under website/static/cookbook-files/monitoring-data-use
and are loaded by path. They use the standard library only.
"""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
SCRIPTS = REPO / "website" / "static" / "cookbook-files" / "monitoring-data-use"


def load_script(name: str):
    spec = importlib.util.spec_from_file_location(name, SCRIPTS / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def harmonize():
    return load_script("harmonize_mentions")


@pytest.fixture(scope="module")
def score():
    return load_script("score_mentions")


@pytest.fixture(scope="module")
def report():
    return load_script("use_report")


def test_harmonization_resolves_variants_and_leaves_external_data(harmonize, tmp_path, capsys):
    out = tmp_path / "harmonized.jsonl"
    assert harmonize.main([str(SCRIPTS / "mentions_example.jsonl"), str(SCRIPTS / "canonical_names.csv"), "-o", str(out)]) == 0
    text = capsys.readouterr().out
    assert "12 mentions: exact 8, contains 2, fuzzy 0, none 2" in text
    rows = [json.loads(line) for line in out.read_text(encoding="utf-8").splitlines()]
    by_text = {r["text"]: r for r in rows}
    assert by_text["HBS"]["dataset_id"] == "EX-HBS-2021" and by_text["HBS"]["match_type"] == "exact"
    assert by_text["the labour survey"]["dataset_id"] == "EX-LFS"
    assert by_text["unemployment rate from the LFS"]["dataset_id"] == "LF_UNEMP_PCT"
    assert by_text["Demographic and Health Survey 2016"]["dataset_id"] is None
    assert by_text["World Development Indicators"]["match_type"] == "none"


def test_match_function_types(harmonize):
    canonical = harmonize.load_canonical(SCRIPTS / "canonical_names.csv")
    assert harmonize.match("Labour Force Survey", canonical, 0.8) == ("EX-LFS", "exact", 1.0)
    dataset, kind, _ = harmonize.match("Labor Force Surveys", canonical, 0.8)
    assert (dataset, kind) == ("EX-LFS", "fuzzy")
    assert harmonize.match("some unrelated phrase", canonical, 0.8) == (None, "none", 0.0)


def test_scoring_against_the_labelled_sample(score, capsys):
    assert score.main([str(SCRIPTS / "mentions_example.jsonl"), str(SCRIPTS / "labelled_sample.csv")]) == 0
    out = capsys.readouterr().out
    assert "labelled mentions 12, extracted 12, correct 11" in out
    assert "precision 0.92  recall 0.92  F1 0.92" in out
    assert "D06: 'food security report'" in out
    assert "D04: 'population and housing census 2014'" in out


def test_use_report_counts_documents_not_mentions(harmonize, report, tmp_path, capsys):
    harmonized = tmp_path / "harmonized.jsonl"
    harmonize.main([str(SCRIPTS / "mentions_example.jsonl"), str(SCRIPTS / "canonical_names.csv"), "-o", str(harmonized)])
    capsys.readouterr()
    assert report.main([str(harmonized), str(SCRIPTS / "documents.csv")]) == 0
    out = capsys.readouterr().out
    assert "6 documents, 12 mentions, 5 datasets of the organization mentioned" in out
    line = next(line for line in out.splitlines() if line.startswith("EX-HBS-2021"))
    assert line.split()[1:3] == ["2", "2"]  # two documents mention it, two use it (two mentions in D01 count once)
    line = next(line for line in out.splitlines() if line.startswith("EX-REP-2025-03"))
    assert line.split()[1:3] == ["2", "0"]  # mentioned in D02 and D06, used in none
    assert "'World Development Indicators': 1 document(s)" in out
