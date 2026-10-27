"""Tests for ai4data.small_ai that need neither a GPU nor a model server."""

import json
from collections import Counter

import pandas as pd
import pytest

from ai4data.small_ai import check, evaluate, generate, loop, plan
from ai4data.small_ai.task import load_task


def make_task(folder, multi_label=False):
    labels = pd.DataFrame(
        [
            {
                "id": "2",
                "name": "Professionals",
                "parent": "",
                "description": "",
                "aliases": "",
            },
            {
                "id": "25",
                "name": "ICT professionals",
                "parent": "2",
                "description": "",
                "aliases": "",
            },
            {
                "id": "2511",
                "name": "Systems analysts",
                "parent": "25",
                "description": "Analyse systems.",
                "aliases": "",
            },
            {
                "id": "2512",
                "name": "Software developers",
                "parent": "25",
                "description": "Write software.",
                "aliases": "Programmers",
            },
            {
                "id": "2513",
                "name": "Web developers",
                "parent": "25",
                "description": "",
                "aliases": "",
            },
            {
                "id": "26",
                "name": "Legal professionals",
                "parent": "2",
                "description": "",
                "aliases": "",
            },
            {
                "id": "2611",
                "name": "Lawyers",
                "parent": "26",
                "description": "",
                "aliases": "",
            },
        ]
    )
    labels.to_csv(folder / "labels.csv", index=False)
    spec = {
        "id": "toy",
        "title": "Toy occupations",
        "statement": "Code a job description.",
        "multi_label": multi_label,
        "levels": [1, 2, 4],
        "generation": {
            "items": "answers",
            "item": "answer",
            "labels_per_request": 2,
            "per_call": 5,
            "group": "siblings",
            "rules": ["Keep answers short."],
        },
        "reference": {
            "calibration": "data/calibration.jsonl",
            "report": "data/report.jsonl",
            "min_support": 1,
        },
        "privacy": {"real_text_to_external": {"orchestrator": False}},
    }
    (folder / "task.json").write_text(json.dumps(spec))
    data = folder / "data"
    data.mkdir()
    rows = [
        {"id": str(i), "text": f"text {i}", "labels": [["2511", "2512", "2611"][i % 3]]}
        for i in range(30)
    ]
    pd.DataFrame(rows).to_json(data / "calibration.jsonl", orient="records", lines=True)
    pd.DataFrame(rows[:9]).to_json(data / "report.jsonl", orient="records", lines=True)
    return load_task(folder)


def test_task_leaves_hierarchy_and_aliases(tmp_path):
    task = make_task(tmp_path)
    assert task.leaves == ["2511", "2512", "2513", "2611"]
    assert task.siblings("2512") == ["2511", "2513"]
    assert task.alias_map()["Programmers"] == "2512"
    assert (
        task.labels["2512"].prompt_line()
        == "2512 Software developers | Write software."
    )


def test_balanced_plan_with_siblings_covers_every_leaf(tmp_path):
    task = make_task(tmp_path)
    siblings = {x: task.siblings(x) for x in task.leaves}
    requests = plan.balanced(task.leaves, 40, 2, siblings=siblings, seed=1)
    counts = plan.offers(requests)
    assert set(counts) == set(task.leaves)
    assert max(counts.values()) - min(counts.values()) <= 3


def test_targeted_plan_is_validated(tmp_path):
    task = make_task(tmp_path)
    known = set(task.leaves)
    ok = plan.targeted(
        [
            {
                "labels": ["2511", "2512"],
                "requests": 3,
                "mode": "contrast",
                "note": "boundary",
            }
        ],
        known,
        "r01",
    )
    assert len(ok) == 3 and ok[0]["mode"] == "contrast"
    with pytest.raises(ValueError, match="unknown labels"):
        plan.targeted([{"labels": ["9999"], "requests": 1}], known, "r01")
    with pytest.raises(ValueError, match="at least two"):
        plan.targeted(
            [{"labels": ["2511"], "requests": 1, "mode": "contrast"}], known, "r01"
        )


def test_check_single_label_drops_several_labels_and_disagreement(tmp_path):
    task = make_task(tmp_path)
    examples = pd.DataFrame(
        [
            {
                "text": "I write code",
                "labels": ["Programmers"],
                "request_id": "a",
                "verified": True,
            },
            {
                "text": "I write code",
                "labels": ["2512"],
                "request_id": "a",
                "verified": True,
            },
            {
                "text": "I analyse and code",
                "labels": ["2511", "2512"],
                "request_id": "a",
                "verified": True,
            },
            {
                "text": "I argue cases",
                "labels": ["2611"],
                "request_id": "b",
                "verified": False,
            },
            {
                "text": "I fix pipes",
                "labels": ["7126"],
                "request_id": "b",
                "verified": None,
            },
        ]
    )
    report, clean, unknown = check.check(examples, task)
    assert report["duplicate texts"] == 1
    assert report["more than one label"] == 1
    assert report["labeller disagreed"] == 1
    assert unknown == Counter({"7126": 1})
    assert clean["labels"].tolist() == [["2512"]]


def test_single_label_evaluation_levels_and_confusions(tmp_path):
    task = make_task(tmp_path)
    gold = pd.DataFrame(
        {"id": ["1", "2", "3", "4"], "labels": [["2511"], ["2512"], ["2611"], ["2512"]]}
    )

    def s(best, other):
        return {
            best: 0.7,
            other: 0.2,
            "2513": 0.05,
            "2611" if best != "2611" else "2511": 0.05,
        }

    pred = [
        {"id": "1", "scores": s("2512", "2511")},  # wrong at 4 digits, right at 2
        {"id": "2", "scores": s("2512", "2511")},
        {"id": "3", "scores": s("2611", "2512")},
        {"id": "4", "scores": s("2511", "2512")},
    ]
    result = evaluate.single_label(task, gold, pred)
    assert result["accuracy"] == 0.5
    assert result["accuracy_by_level"] == {"1": 1.0, "2": 1.0, "4": 0.5}
    assert result["top3_accuracy"] == 1.0
    pairs = {(c["reference"], c["predicted"]) for c in result["confusions"]}
    assert pairs == {("2511", "2512"), ("2512", "2511")}
    gap = evaluate.gap_report(task, result, Counter({"2512": 10}))
    assert gap["headline"] == 0.5
    assert gap["thin_labels"][0]["train_examples"] == 0


def test_multi_label_matches_the_cookbook_definition():
    gold = pd.DataFrame({"id": ["a", "b"], "labels": [["X", "Y"], ["Y"]]})
    pred = [
        {"id": "a", "scores": {"X": 0.9, "Y": 0.1, "Z": 0.6}},
        {"id": "b", "scores": {"X": 0.1, "Y": 0.8, "Z": 0.0}},
    ]
    result = evaluate.multi_label(gold, pred, 0.5, 5)
    # measured labels X and Y; predicted a: X, b: Y; Z is not measured
    assert result["micro_precision"] == 1.0
    assert result["micro_recall"] == pytest.approx(2 / 3, abs=1e-4)


def test_writer_prompt_carries_contrast_and_note(tmp_path):
    task = make_task(tmp_path)
    text = generate.writer_prompt(
        task,
        {
            "request_id": "r",
            "labels": ["2511", "2512"],
            "mode": "contrast",
            "note": "developers who also design systems",
        },
        5,
    )
    assert "exactly one" in text and "close to the boundary" in text
    assert "developers who also design systems" in text
    assert "2512 Software developers | Write software." in text


def test_run_plans_and_reference_split(tmp_path):
    task_dir = tmp_path / "task"
    task_dir.mkdir()
    make_task(task_dir)
    run = loop.init_run(
        task_dir,
        tmp_path / "run",
        "targeted",
        baseline_requests=6,
        round_requests=4,
        rounds=2,
    )
    feedback, threshold = run.reference("feedback"), run.reference("threshold")
    assert len(feedback) + len(threshold) == 30 and len(feedback) > len(threshold) > 0
    assert len(run.plan_for(0, None)) == 6
    with pytest.raises(ValueError, match="exactly 4"):
        run.plan_for(
            1,
            [
                {
                    "labels": ["2511"],
                    "requests": 3,
                    "mode": "positive",
                    "note": "more examples",
                }
            ],
        )
    with pytest.raises(ValueError, match="background"):
        run.plan_for(
            1,
            [
                {
                    "labels": [],
                    "requests": 4,
                    "mode": "background",
                    "note": "none of them",
                }
            ],
        )
    assert (
        len(
            run.plan_for(
                1,
                [
                    {
                        "labels": ["2511"],
                        "requests": 4,
                        "mode": "positive",
                        "note": "weak",
                    }
                ],
            )
        )
        == 4
    )


def test_agreement_table_counts_writer_labels(tmp_path):
    examples = pd.DataFrame(
        [
            {
                "text": "a",
                "labels": ["2512"],
                "written_labels": ["2512"],
                "verified": True,
                "kind": "topic",
            },
            {
                "text": "b",
                "labels": ["2512"],
                "written_labels": ["2512"],
                "verified": False,
                "kind": "topic",
            },
            {
                "text": "c",
                "labels": ["2611"],
                "written_labels": ["2611"],
                "verified": True,
                "kind": "topic",
            },
        ]
    )
    table = check.agreement(examples).set_index("label")
    assert table.loc["2512", "agreement"] == 0.5
    assert table.loc["2611", "agreement"] == 1.0


def test_status_counts_planned_rounds(tmp_path):
    task_dir = tmp_path / "task"
    task_dir.mkdir()
    make_task(task_dir)
    run = loop.init_run(
        task_dir,
        tmp_path / "run",
        "control",
        baseline_requests=6,
        round_requests=4,
        rounds=3,
    )
    status = run.status()
    assert status["planned_rounds_remaining"] == 3 and status["rounds_done"] == 0
