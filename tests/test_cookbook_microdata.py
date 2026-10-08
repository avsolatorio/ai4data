"""Tests for the scripts shipped with the microdata documentation cookbook.

The scripts live under website/static/cookbook-files/microdata-documentation
and are loaded by path. The conversion test validates its output with the
metadata checker from the dissemination cookbook, against the pinned schema
fixtures in tests/fixtures/wb-schemas.
"""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
SCRIPTS = REPO / "website" / "static" / "cookbook-files" / "microdata-documentation"
DISSEMINATION = REPO / "website" / "static" / "cookbook-files" / "ai-ready-dissemination"
SCHEMAS = REPO / "tests" / "fixtures" / "wb-schemas"


def load_script(folder: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, folder / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def check():
    return load_script(SCRIPTS, "check_dictionary")


@pytest.fixture(scope="module")
def convert():
    return load_script(SCRIPTS, "dictionary_to_ddi")


@pytest.fixture(scope="module")
def score():
    return load_script(SCRIPTS, "score_variable_search")


def test_dictionary_check_finds_the_planted_defects(check, capsys):
    assert check.main([str(SCRIPTS / "lfs_2025q2_dictionary.csv")]) == 1
    out = capsys.readouterr().out
    assert "4 error(s)" in out
    for expected in (
        "F1/hours: name appears 2 times",
        "F1/relationship: categorical variable without value labels",
        "F1/informal: label repeats the name",
        "F1/q17b: no label",
        "F1/income: numeric variable without a missing-value statement",
    ):
        assert expected in out


def test_dictionary_check_passes_a_clean_dictionary(check, capsys, tmp_path):
    rows = (SCRIPTS / "lfs_2025q2_dictionary.csv").read_text(encoding="utf-8").splitlines()
    clean = [rows[0]] + [r for r in rows[1:] if not r.startswith(("F1,hours,Hours worked last week in main", "F1,q17b,"))]
    clean = [
        r.replace("F1,informal,informal,", "F1,informal,Informal employment in main job,")
        .replace("Q5. What is [NAME]'s relationship to the head of household?,,", "Q5. What is [NAME]'s relationship to the head of household?,1=Head;2=Spouse;3=Child;4=Other relative;5=Not related,")
        for r in clean
    ]
    path = tmp_path / "clean.csv"
    path.write_text("\n".join(clean) + "\n", encoding="utf-8")
    assert check.main([str(path)]) == 0
    assert "0 error(s)" in capsys.readouterr().out


def test_conversion_produces_a_valid_and_complete_study_record(convert, tmp_path, capsys):
    pytest.importorskip("jsonschema")
    out = tmp_path / "study.json"
    assert convert.main([
        str(SCRIPTS / "lfs_2025q2_dictionary.csv"),
        "--study",
        str(DISSEMINATION / "example_microdata.json"),
        "-o",
        str(out),
    ]) == 0
    record = json.loads(out.read_text(encoding="utf-8"))
    assert len(record["variables"]) == 18
    assert record["data_files"] == [{"file_id": "F1", "file_name": "F1.csv", "var_count": 18}]
    educ = next(v for v in record["variables"] if v["name"] == "educ")
    assert educ["labl"] == "Highest level of education completed"
    assert {"value": "5", "label": "Tertiary"} in educ["var_catgry"]
    assert {"value": "-9", "label": "Missing"} in educ["var_catgry"]
    assert educ["var_concept"] == [{"title": "ISCED 2011 level"}]

    checker = load_script(DISSEMINATION, "check_metadata")
    code = checker.main([
        str(out),
        str(DISSEMINATION / "profile_microdata.json"),
        "--level",
        "ai-ready",
        "--schema-dir",
        str(SCHEMAS),
    ])
    text = capsys.readouterr().out
    assert "1 of 1 records are valid instances of the schema" in text
    assert "1 of 1 records complete" in text
    assert code == 0


def test_variable_search_baseline_matches_the_cookbook_text(score, capsys):
    assert score.main([str(SCRIPTS / "variable_questions.csv"), str(SCRIPTS / "lfs_2025q2_dictionary.csv")]) == 0
    out = capsys.readouterr().out
    assert "en      10   0.80   0.72" in out
    assert "fr       2   0.00   0.00" in out


def test_variable_search_ranks_by_label_question_and_concept(score):
    dictionary = score.load_csv(SCRIPTS / "lfs_2025q2_dictionary.csv")
    assert score.search("highest education level", dictionary)[0] == "educ"
    assert score.search("survey weight", dictionary)[0] == "weight"
    assert score.search("quarterly inflation", dictionary) == []
