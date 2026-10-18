"""synth/models.py registry, llm.chat request fields and the judge panel's kappa."""
import importlib.util
import json
import sys
from pathlib import Path

from relevance_explainer.synth import llm
from relevance_explainer.synth.models import MODELS

ROOT = Path(__file__).resolve().parents[1]


def test_registry_entries_complete():
    ports = [m["port"] for m in MODELS.values()]
    assert len(ports) == len(set(ports))
    for name, m in MODELS.items():
        assert {"repo", "family", "licence", "port", "urls", "request", "serve", "mem", "role"} <= set(m), name
        assert all(u.startswith("http://localhost:") for u in m["urls"])


def test_existing_teachers_unchanged():
    # The data scripts name these two directly; the registry must agree with them.
    from relevance_explainer.synth.challenge import GEMMA, TEACHER
    assert (TEACHER[0], TEACHER[1]) == ("qwen3.8-27b", MODELS["qwen3.8-27b"]["urls"])
    assert (GEMMA[0], GEMMA[1]) == ("gemma-4-31b", MODELS["gemma-4-31b"]["urls"])


def test_chat_extra_fields(monkeypatch):
    sent = {}

    class R:
        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

    def fake(req, timeout):
        sent.update(json.loads(req.data))
        return R()

    monkeypatch.setattr(llm, "_inflight", {})  # the load balancer's counts are shared with other tests
    monkeypatch.setattr(llm.urllib.request, "urlopen", fake)
    monkeypatch.setattr(llm.json, "load", lambda r: {"choices": [{"message": {"content": "{}"}, "finish_reason": "stop"}]})
    monkeypatch.setenv("LLM_EXTRA", '{"chat_template_kwargs": {"enable_thinking": false}}')
    llm.chat("m", "s", "u", base_urls=["http://localhost:1/v1"], extra={"reasoning_effort": "low"})
    assert sent["reasoning_effort"] == "low" and sent["chat_template_kwargs"] == {"enable_thinking": False}
    sent.clear()
    llm.chat("m", "s", "u", base_urls=["http://localhost:1/v1"])
    assert "reasoning_effort" not in sent


def test_kappa():
    spec = importlib.util.spec_from_file_location("judge_panel", ROOT / "scripts" / "judge_panel.py")
    jp = importlib.util.module_from_spec(spec)
    sys.argv = ["judge_panel"]
    spec.loader.exec_module(jp)
    a = ["relevant", "irrelevant", "partly_relevant", "relevant"]
    assert jp.kappa(a, a) == 1.0
    assert abs(jp.kappa(["relevant", "irrelevant"] * 2, ["relevant", "relevant", "irrelevant", "irrelevant"])) < 1e-9
