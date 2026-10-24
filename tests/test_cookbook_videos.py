"""Tests for the scripts shipped with the explainer-videos cookbook."""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
SCRIPTS = REPO / "website" / "static" / "cookbook-files" / "explainer-videos"


def load_script(name: str):
    spec = importlib.util.spec_from_file_location(name, SCRIPTS / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def scripts():
    names = ["check_narration", "check_captions", "video_record"]
    return {n: load_script(n) for n in names}


def corrected_narration(tmp_path: Path) -> Path:
    data = json.loads((SCRIPTS / "narration.json").read_text(encoding="utf-8"))
    for step in data["steps"]:
        if "16.3" in step.get("say", ""):
            step["say"] = step["say"].replace("16.3", "16.8")
            step["speak"] = step["speak"].replace(
                "sixteen point three", "sixteen point eight"
            )
        if "speak" in step:
            step["speak"] = step["speak"].replace("61.2", "sixty-one point two")
        step["say"] = step.get("say", "").replace(
            ", which the methodology note of the survey describes in full.",
            ". The methodology note of the survey describes them in full.",
        )
    path = tmp_path / "narration.json"
    path.write_text(json.dumps(data), encoding="utf-8")
    return path


def test_narration_finds_the_planted_errors(scripts, capsys):
    rc = scripts["check_narration"].main(
        [str(SCRIPTS / "narration.json"), str(SCRIPTS / "release_table.csv")]
    )
    out = capsys.readouterr().out
    assert rc == 1
    assert "28 narrated steps, 555 spoken words" in out
    assert "estimated length: 3 min 27 s" in out
    assert "16.3 is not a value or a previous value in release_table.csv" in out
    assert "speak keeps the digits 61.2" in out
    assert "sentence of 32 words (limit 25)" in out
    assert out.count("problem:") == 3


def test_narration_passes_after_correction(scripts, capsys, tmp_path):
    rc = scripts["check_narration"].main(
        [str(corrected_narration(tmp_path)), str(SCRIPTS / "release_table.csv")]
    )
    out = capsys.readouterr().out
    assert rc == 0, out
    assert "figures in say: 11; 11 in the table, 0 not" in out


def test_narration_flags_digits_without_speak(scripts, capsys, tmp_path):
    data = {"steps": [{"say": "The rate was 6.1 percent."}]}
    path = tmp_path / "short.json"
    path.write_text(json.dumps(data), encoding="utf-8")
    rc = scripts["check_narration"].main(
        [str(path), str(SCRIPTS / "release_table.csv")]
    )
    out = capsys.readouterr().out
    assert rc == 1
    assert "say has digits and the step has no speak" in out
    assert "estimated length is outside the target range" in out


def test_captions_find_the_planted_problems(scripts, capsys):
    rc = scripts["check_captions"].main([str(SCRIPTS / "captions.vtt")])
    out = capsys.readouterr().out
    assert rc == 1
    assert "60 cues" in out
    assert "cue 46 (00:02:39.370): line of 68 characters (limit 37)" in out
    assert "cue 47 (00:02:43.000): starts before the previous cue ends" in out
    assert out.count("problem:") == 2


def test_captions_out_of_order_and_header(scripts, capsys, tmp_path):
    path = tmp_path / "bad.vtt"
    path.write_text(
        "00:00:05.000 --> 00:00:07.000\nSecond.\n\n"
        "00:00:01.000 --> 00:00:03.000\nFirst.\n\n"
        "00:00:08.000 --> 00:00:08.000\nEmpty.\n",
        encoding="utf-8",
    )
    assert scripts["check_captions"].main([str(path)]) == 1
    out = capsys.readouterr().out
    assert "the file does not start with WEBVTT" in out
    assert "cue 2 (00:00:01.000): starts before the previous cue" in out
    assert "not after it starts" in out


def test_captions_rewrap_one_cue_per_step(scripts, capsys, tmp_path):
    long = tmp_path / "tool.vtt"
    long.write_text(
        "WEBVTT\n\n00:00:01.000 --> 00:00:09.000\n"
        "The unemployment rate was 6.1 percent in the second quarter, after"
        " 6.3 percent in the first. The second-quarter figure is provisional.\n",
        encoding="utf-8",
    )
    assert scripts["check_captions"].main([str(long)]) == 1
    capsys.readouterr()
    out_file = tmp_path / "wrapped.vtt"
    assert scripts["check_captions"].main([str(long), "--rewrap", str(out_file)]) == 0
    assert scripts["check_captions"].main([str(out_file)]) == 0
    out = capsys.readouterr().out
    assert "result: PASS" in out
    _, cues = scripts["check_captions"].parse(out_file.read_text(encoding="utf-8"))
    assert cues[0].start == 1.0 and abs(cues[-1].end - 9.0) < 1e-6
    words = " ".join(" ".join(c.lines) for c in cues).split()
    assert words[0] == "The" and words[-1] == "provisional."


def test_video_record(scripts, capsys, tmp_path):
    out_file = tmp_path / "record.jsonld"
    rc = scripts["video_record"].main(
        [str(SCRIPTS / "video_card.json"), "--out", str(out_file)]
    )
    out = capsys.readouterr().out
    assert rc == 0
    assert "duration PT3M34S, 2 sources" in out
    assert "duration taken from the last caption cue" in out
    record = json.loads(out_file.read_text(encoding="utf-8"))
    assert record["@type"] == "VideoObject"
    assert record["digitalSourceType"].endswith("CompositeSyntheticDigitalSource")
    assert record["accessibilityFeature"] == ["closedCaptions", "transcript"]
    assert "synthetic voice" in record["description"]
    assert [s["identifier"] for s in record["isBasedOn"]] == [
        "EX-LFS-2025-Q2",
        "EX-REP-2025-02",
    ]
    assert record["transcript"].startswith("This video explains")


def test_video_record_requires_disclosure(scripts, capsys, tmp_path):
    card = json.loads((SCRIPTS / "video_card.json").read_text(encoding="utf-8"))
    card["narration_disclosure"] = "Narrated by the office."
    card["duration_seconds"] = 221
    (tmp_path / "captions.vtt").write_text(
        (SCRIPTS / "captions.vtt").read_text(encoding="utf-8"), encoding="utf-8"
    )
    path = tmp_path / "card.json"
    path.write_text(json.dumps(card), encoding="utf-8")
    assert scripts["video_record"].main([str(path)]) == 1
    assert "disclosure does not say so" in capsys.readouterr().out
    card["narration_disclosure"] = "The narration uses a synthetic voice."
    path.write_text(json.dumps(card), encoding="utf-8")
    assert scripts["video_record"].main([str(path)]) == 0
    out = capsys.readouterr().out
    assert '"duration": "PT3M41S"' in out
    assert "note:" not in out
