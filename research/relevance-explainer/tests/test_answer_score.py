import math
import json

import torch

from relevance_explainer.tiny import selfcheck
from relevance_explainer.tiny.answer_score import ScoreHead, features
from relevance_explainer.tiny.model import Config, TinyLM
from relevance_explainer.viewer import stream

NL, VFIRST = 5, [6, 7, 8]


def _model():
    torch.manual_seed(0)
    return TinyLM(Config(vocab=60, d_model=32, n_layer=2, n_head=2, d_ff=64, max_len=128, dropout=0.0)).eval()


def _decode(m, prompt, temperature=0.0, seed=0):
    trace, out, first = [], [], None
    for k, (tok, logits) in enumerate(stream(m, prompt, max_new=60, no_repeat=4, temperature=temperature, seed=seed, trace=trace)):
        if k == 0:
            first = logits
        if tok == 1:
            break
        out.append(tok)
    return out, trace, first


def test_decoding_trace_features_equal_teacher_forced_features():
    """The app scores answers from statistics recorded while decoding; they must equal the features the head was
    trained on (one teacher-forced pass in selfcheck.output_features), with and without a forced verdict line."""
    m = _model()
    selfcheck.tk_newline = NL
    prompt = torch.randint(10, 60, (12,)).tolist()
    _, _, first = _decode(m, prompt)
    vp = torch.softmax(first.float()[VFIRST], 0).max().item()  # from the unforced first step, as in the app
    ended_cases = 0
    for forced in ([], [6, NL]):
        for temperature, seed in [(0.0, 0)] + [(0.8, s) for s in range(8)]:
            gen, trace, _ = _decode(m, prompt + forced, temperature, seed)
            out = forced + gen
            ref = selfcheck.output_features(m, prompt, out, VFIRST)["scalars"]
            got = features(trace, out, NL, vp, False, len(forced))
            assert len(got) == len(ref) + 1
            # at the length limit no step follows the last token, so its end-of-text probability is not available
            ended = len(trace) > len(gen)
            same = [abs(a - b) < 1e-4 for a, b in zip(got, ref)]
            assert all(same[:6]) and (same[6] or not ended), (forced, temperature, seed, got, ref)
            ended_cases += ended
    assert ended_cases > 0  # the end-of-text probability was compared at least once


def test_score_head_is_a_logistic_of_standardized_features(tmp_path):
    p = tmp_path / "h.json"
    p.write_text(json.dumps({"mean": [1.0, 0.0], "std": [2.0, 1.0], "weight": [1.0, -1.0], "bias": 0.5}))
    h = ScoreHead(p)
    z = 0.5 + (3.0 - 1.0) / 2.0 - 1.0 * 0.5
    assert abs(h([3.0, 0.5]) - 1 / (1 + math.exp(-z))) < 1e-12
