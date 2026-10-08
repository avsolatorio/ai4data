"""Tests for the scripts shipped with the metadata-curation-with-llms cookbook."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
SCRIPTS = REPO / "website" / "static" / "cookbook-files" / "metadata-curation-with-llms"


def load_script(name: str):
    spec = importlib.util.spec_from_file_location(name, SCRIPTS / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def score():
    return load_script("score_suggestions")


@pytest.fixture(scope="module")
def agreement():
    return load_script("assess_agreement")


@pytest.fixture(scope="module")
def vocab():
    return load_script("map_vocabulary")


def test_acceptance_rates_per_field_and_source(score, capsys):
    assert score.main([str(SCRIPTS / "suggestions_example.jsonl"), str(SCRIPTS / "decisions_example.csv")]) == 0
    out = capsys.readouterr().out
    assert "10 suggestions, 10 decided, 0 undecided" in out
    assert "measurement_unit" in out and "rejected, with reasons:" in out
    assert "HE_U5_MORT_RATE/measurement_unit (reviewer)" in out
    field_lines = out.split("by field")[1].split("by source")[0].splitlines()
    keywords = next(line for line in field_lines if line.startswith("keywords"))
    assert keywords.split()[-1] == "1.00"


def test_agreement_flags_the_lenient_dimension(agreement, capsys):
    assert agreement.main([str(SCRIPTS / "model_scores.csv"), str(SCRIPTS / "curator_scores.csv")]) == 0
    out = capsys.readouterr().out
    assert "5 records, 4 dimensions" in out
    consistency = next(line for line in out.splitlines() if line.startswith("consistency"))
    assert "model more lenient" in consistency
    assert "1 dimension(s) miscalibrated" in out


def test_vocabulary_mapping_counts(vocab, capsys):
    assert vocab.main([str(SCRIPTS / "keywords_freetext.csv"), str(SCRIPTS / "vocabulary.csv")]) == 0
    out = capsys.readouterr().out
    assert "14 keywords: exact 10, fuzzy 2, none 2" in out
    assert "FS_UNDERNOURISH_PCT: 'hunger'" in out
    assert "LF_UNEMP_PCT: 'jobless'" in out
    assert "'unemployment rates'" in out and "-> Unemployment" in out
