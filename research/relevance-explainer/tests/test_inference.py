import io
import json
import re
from contextlib import redirect_stdout
from pathlib import Path

import pytest

from relevance_explainer import inference

ROOT = Path(__file__).resolve().parents[1]
RUN = ROOT / "runs" / "final-C-dpo"
HAVE_RUN = all((RUN / f).exists() for f in inference.FILES)
PARITY = RUN / "cpu_answers_itemfirst_nosim_v4.jsonl"


def test_render_matches_the_app():
    """inference.render gives the input the app sends to the claim-carrying model for realistic examples."""
    from relevance_explainer.webapp.server import _REALISTIC, render_for
    ids = sorted(_REALISTIC)[:25]
    assert ids, "app/examples_realistic.json has no examples"
    for i in ids:
        e = _REALISTIC[i]
        app = re.sub(r"\nRetrieval similarity: [-0-9.]+", "", render_for(i, None)["itemfirst"])
        assert inference.render(e["record"], e["query"], e["language"], e.get("facets")) == app, i


@pytest.fixture(scope="module")
def bundle(tmp_path_factory):
    if not HAVE_RUN:
        pytest.skip("trained run runs/final-C-dpo is not present")
    out = tmp_path_factory.mktemp("bundle")
    inference.build_bundle(RUN, out, "test", "0")
    return out


def test_bundle_reproduces_the_app_answers(bundle):
    """Same CPU path as scripts/cpu_parity_c.py (4 threads): identical text, checks and scores."""
    if not PARITY.exists():
        pytest.skip("no CPU parity answers to compare with")
    ex = inference.Explainer(bundle, threads=4)
    rows = [json.loads(l) for l in open(PARITY)][:4]
    for r in rows:
        o = ex.explain_text(r["input"])
        assert o["raw"] == r["text"]
        assert o["verdict"] == r["verdict"] and o["all_verified"] == r["verified"]
        assert o["probabilities"] == r["probs"] and o["verdict_set"] == r["verdict_set"]
        assert o["expected_grade"] == r["expected_grade"] and o["faithfulness"] == r["faith"]


def test_modified_bundle_is_refused(bundle, tmp_path):
    import shutil
    bad = tmp_path / "bad"
    shutil.copytree(bundle, bad)
    cal = json.loads((bad / "calibration.json").read_text())
    cal["temperature"] = 2.0
    (bad / "calibration.json").write_text(json.dumps(cal))
    with pytest.raises(ValueError, match="calibration.json"):
        inference.Explainer(bad)


def test_command_line_explains_jsonl(bundle, tmp_path):
    req = tmp_path / "req.jsonl"
    from relevance_explainer.webapp.server import _REALISTIC
    e = _REALISTIC[sorted(_REALISTIC)[0]]
    req.write_text(json.dumps({"record": e["record"], "search": e["query"], "language": e["language"],
                               "facets": e.get("facets")}) + "\n")
    buf = io.StringIO()
    with redirect_stdout(buf):
        inference.main(["explain", "--bundle", str(bundle), "--input", str(req)])
    o = json.loads(buf.getvalue().strip().split("\n")[-1])
    assert o["verdict"] in ("relevant", "partly_relevant", "irrelevant")
    assert o["explanation"] and o["input"].startswith("Item type:") and o["model"] == "test"
