"""Tests for the scripts shipped with the small-ai-from-synthetic-data cookbook.

The scripts that train or run the model need torch and transformers and
are not tested here; the tests cover the scripts that prepare, plan,
check, split, combine, and evaluate, on the example files.
"""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pandas as pd
import pytest

REPO = Path(__file__).resolve().parents[1]
SCRIPTS = (
    REPO / "website" / "static" / "cookbook-files" / "small-ai-from-synthetic-data"
)
sys.path.insert(0, str(SCRIPTS))


def load_script(name: str):
    spec = importlib.util.spec_from_file_location(name, SCRIPTS / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def test_prepare_labels(tmp_path, capsys):
    module = load_script("prepare_labels")
    out = tmp_path / "labels.csv"
    assert (
        module.main(
            [
                str(SCRIPTS / "wb_topics.json"),
                str(SCRIPTS / "label_merges.csv"),
                str(out),
            ]
        )
        == 0
    )
    printed = capsys.readouterr().out
    assert (
        "entries on the list   95" in printed and "labels                86" in printed
    )
    labels = pd.read_csv(out, keep_default_na=False)
    assert labels["label"].is_unique and len(labels) == 86
    assert "Jobs & Development" not in set(labels["label"])
    skills = labels.set_index("label").loc["Skills Development", "parents"]
    assert skills == "Education; Jobs and Development; Social Protection"
    assert out.read_text(encoding="utf-8") == (SCRIPTS / "labels.csv").read_text(
        encoding="utf-8"
    )


def test_prepare_labels_rejects_unknown_merge(tmp_path):
    module = load_script("prepare_labels")
    merges = tmp_path / "merges.csv"
    merges.write_text("from,to,reason\nNo Such Topic,Energy,test\n", encoding="utf-8")
    assert (
        module.main(
            [str(SCRIPTS / "wb_topics.json"), str(merges), str(tmp_path / "x.csv")]
        )
        == 1
    )


def test_plan_requests(tmp_path, capsys):
    module = load_script("plan_requests")
    out = tmp_path / "requests.jsonl"
    assert (
        module.main(
            [
                str(SCRIPTS / "labels.csv"),
                str(out),
                "--requests",
                "1000",
                "--per-request",
                "8",
            ]
        )
        == 0
    )
    printed = capsys.readouterr().out
    assert "least offered  92" in printed and "most offered   95" in printed
    assert (
        module.main(
            [str(SCRIPTS / "labels.csv"), str(out), "--requests", "1000", "--uniform"]
        )
        == 0
    )
    printed = capsys.readouterr().out
    assert "least offered  75" in printed and "most offered   114" in printed
    assert (
        module.main(
            [
                str(SCRIPTS / "labels.csv"),
                str(out),
                "--requests",
                "3",
                "--background",
                "2",
            ]
        )
        == 0
    )
    plans = pd.read_json(out, lines=True)
    assert list(plans["request_id"]) == [
        "req-00000",
        "req-00001",
        "req-00002",
        "bg-00000",
        "bg-00001",
    ]
    assert plans["labels"].map(len).tolist() == [8, 8, 8, 0, 0]


def test_check_examples(tmp_path, capsys):
    module = load_script("check_examples")
    out = tmp_path / "clean.jsonl"
    assert (
        module.main(
            [
                str(SCRIPTS / "examples_sample.jsonl"),
                str(SCRIPTS / "labels.csv"),
                "--out",
                str(out),
            ]
        )
        == 0
    )
    printed = capsys.readouterr().out
    for line in (
        "unknown label names        7",
        "duplicate texts            3",
        "clean examples             37",
    ):
        assert line in printed
    clean = pd.read_json(out, lines=True)
    known = set(pd.read_csv(SCRIPTS / "labels.csv")["label"])
    assert all(set(labels) <= known and labels for labels in clean["labels"])
    assert clean["text"].str.lower().is_unique


def test_check_examples_keeps_background(capsys):
    module = load_script("check_examples")
    known, merged = module.label_set(str(SCRIPTS / "labels.csv"))
    examples = pd.DataFrame(
        {
            "text": [
                "The sample covers 1,200 households.",
                "Clean water reached the district.",
                "No label here.",
            ],
            "labels": [[], ["Water Supply"], []],
            "request_id": ["bg-1", "req-1", "req-2"],
            "generator": ["m", "m", "m"],
            "kind": ["background", "topic", "topic"],
        }
    )
    report, clean, _ = module.check(examples, known, merged)
    assert report["background"] == 1 and report["no labels"] == 1
    assert list(clean["request_id"]) == ["bg-1", "req-1"]


def test_split_examples_keeps_requests_whole(tmp_path, capsys):
    module = load_script("split_examples")
    rows = [
        {"text": f"s{r}-{i}", "labels": ["Energy"], "request_id": f"req-{r}"}
        for r in range(40)
        for i in range(5)
    ]
    examples = tmp_path / "examples.jsonl"
    pd.DataFrame(rows).to_json(examples, orient="records", lines=True)
    assert module.main([str(examples), str(tmp_path / "splits")]) == 0
    parts = {
        n: pd.read_json(tmp_path / "splits" / f"{n}.jsonl", lines=True)
        for n in ("train", "dev", "test")
    }
    assert sum(len(p) for p in parts.values()) == 200
    ids = [set(p["request_id"]) for p in parts.values()]
    assert not (ids[0] & ids[1] or ids[0] & ids[2] or ids[1] & ids[2])


def test_topic_scores():
    module = load_script("topic_scores")
    text = "Marriage reduces female employment. It is short. The paper studies South Asia in detail."
    assert module.split_sentences(text) == [
        "Marriage reduces female employment.",
        "The paper studies South Asia in detail.",
    ]
    assert module.split_sentences("Too short.") == ["Too short."]
    rows = [[0.1, 0.9], [0.3, 0.1]]
    assert module.aggregate(rows, "mean").round(2).tolist() == [0.2, 0.5]
    assert module.aggregate(rows, "max").tolist() == [0.3, 0.9]
    with pytest.raises(ValueError):
        module.aggregate(rows, "median")
    assert module.choose({"A": 0.5, "B": 0.2, "C": 0.01}, 0.1) == ["A", "B"]
    assert module.choose({"A": 0.5, "B": 0.2, "C": 0.3}, 0.1, 2) == ["A", "C"]


def test_evaluate(tmp_path, capsys):
    module = load_script("evaluate")
    gold = tmp_path / "gold.jsonl"
    scores = tmp_path / "scores.jsonl"
    pd.DataFrame(
        {"id": ["1", "2"], "labels": [["Energy"], ["Health", "Gender"]]}
    ).to_json(gold, orient="records", lines=True)
    with scores.open("w", encoding="utf-8") as handle:
        handle.write(
            json.dumps(
                {"id": "1", "scores": {"Energy": 0.9, "Health": 0.05, "Gender": 0.0}}
            )
            + "\n"
        )
        handle.write(
            json.dumps(
                {"id": "2", "scores": {"Energy": 0.6, "Health": 0.7, "Gender": 0.1}}
            )
            + "\n"
        )
    assert (
        module.main([str(gold), str(scores), "--threshold", "0.5", "--top", "5"]) == 0
    )
    printed = capsys.readouterr().out
    assert "micro precision       0.667" in printed
    assert "micro recall          0.667" in printed
    assert module.main([str(gold), str(scores), "--sweep"]) == 0
    assert "threshold 0.02" in capsys.readouterr().out


def test_build_real_testset_mapping():
    pytest.importorskip("requests")
    module = load_script("build_real_testset")
    crosswalk = pd.read_csv(SCRIPTS / "crosswalk.csv")
    words = " ".join(["word"] * 45)
    documents = [
        {
            "id": "2",
            "display_title": "A",
            "abstracts": {"cdata!": words},
            "subtopic": "Water Supply and Sanitation,Hydrology",
        },
        {
            "id": "1",
            "display_title": "B",
            "abstracts": words,
            "teratopic": "Disability",
        },
        {
            "id": "3",
            "display_title": "C",
            "abstracts": "too short",
            "subtopic": "Education",
        },
    ]
    testset = module.to_testset(documents, crosswalk)
    assert testset["id"].tolist() == ["2"]
    assert testset.loc[0, "labels"] == ["Sanitation", "Water Supply"]
    assert testset.loc[0, "source_topics"] == ["Water Supply and Sanitation"]


def test_real_sample_matches_crosswalk():
    known = set(pd.read_csv(SCRIPTS / "labels.csv")["label"])
    crosswalk = pd.read_csv(SCRIPTS / "crosswalk.csv")
    assert set(crosswalk["label"]) <= known
    mapping = crosswalk.groupby("source_topic")["label"].apply(set).to_dict()
    sample = pd.read_json(SCRIPTS / "real_sample.jsonl", lines=True)
    for labels, topics in zip(sample["labels"], sample["source_topics"]):
        assert set(labels) == set().union(*(mapping[t] for t in topics))


def test_import_hub_dataset_matches_sentences():
    module = load_script("import_hub_dataset")
    raw = pd.DataFrame(
        {
            "content": [
                '[{"sentence": "Roads cut travel time.", "topics": ["Transport"]}]',
                '{"sentence": "Clinics opened."}',
            ],
            "id": ["run-1", None],
        }
    )
    processed = pd.DataFrame(
        {
            "sentence": [
                "Roads cut travel time.",
                "Clinics opened.",
                "Not in any reply.",
            ],
            "topics": [["Transport"], ["Health"], None],
            "model_name": ["m", "m", "m"],
        }
    )
    examples = module.to_examples(processed, raw)
    assert examples["request_id"].tolist() == ["run-1", "response-00001", "unmatched"]
    assert examples["labels"].tolist() == [["Transport"], ["Health"], []]


def test_generate_examples_parse_and_schema():
    pytest.importorskip("openai")
    module = load_script("generate_examples")
    content = json.dumps(
        {
            "examples": [
                {"text": "  Roads   cut time. ", "labels": ["Transport", "Energy"]},
                {"text": "x", "labels": ["Nope"]},
            ]
        }
    )
    rows = module.parse(
        {"request_id": "req-1", "model": "m", "content": content}, ["Transport"]
    )
    assert rows == [
        {
            "text": "Roads cut time.",
            "labels": ["Transport"],
            "request_id": "req-1",
            "generator": "m",
            "kind": "topic",
        }
    ]
    background = module.parse(
        {
            "request_id": "bg-1",
            "model": "m",
            "content": json.dumps({"examples": [{"text": "The data cover 2010."}]}),
        },
        [],
    )
    assert background[0]["kind"] == "background" and background[0]["labels"] == []
    enum = module.schema(["Transport"], 20)["schema"]["properties"]["examples"][
        "items"
    ]["properties"]["labels"]["items"]["enum"]
    assert enum == ["Transport"]
    assert (
        "labels"
        not in module.schema([], 20)["schema"]["properties"]["examples"]["items"][
            "properties"
        ]
    )
    assert (
        module.parse(
            {"request_id": "r", "model": "m", "content": "not json"}, ["Transport"]
        )
        == []
    )
