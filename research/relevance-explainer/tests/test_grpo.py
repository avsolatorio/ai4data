from relevance_explainer.tiny.grpo import FAIL, advantages, pack, reward

GOOD = {"faithful": 5, "coherent": 5, "consistent": True, "language_ok": True}


def test_reward_scale_and_penalties():
    assert reward(GOOD, "Covers Kenya.", "Kenya", False, 0.25) == 2.5
    bad = {**GOOD, "faithful": 1, "coherent": 1, "consistent": False, "language_ok": False}
    assert reward(bad, "x", "x", False, 0.25) == -2.0
    assert reward(GOOD, "Covers Ghana in 2015.", "Kenya", False, 0.25) == 2.0  # two unsupported items
    assert reward(None, "", "x", False, 0.25) == FAIL
    assert reward(GOOD, "text", "x", True, 0.25) == FAIL
    assert reward(None, "text", "x", False, 0.25) is None


def test_informativeness_term_and_prompt():
    from relevance_explainer.tiny.grpo import INFORM_SYSTEM
    s = {**GOOD, "informative": 1}
    assert reward(s, "Covers Kenya.", "Kenya", False, 0.25, 1.0) == 2.5
    assert reward({**s, "informative": 5}, "Covers Kenya.", "Kenya", False, 0.25, 1.0) == 3.5
    assert reward({**s, "informative": 5}, "Covers Kenya.", "Kenya", False, 0.25, 0.0) == 2.5
    assert "informative (1-5)" in INFORM_SYSTEM and '"informative": int}' in INFORM_SYSTEM


def test_advantages_standardize_and_skip():
    a = advantages([[1.0, 3.0], [2.0, 2.0], [1.0, None], [0.0, None, 2.0]])
    assert abs(a[0][0] + 1) < 1e-3 and abs(a[0][1] - 1) < 1e-3
    assert a[1] == [None, None]
    assert a[2] == [None, None]
    assert a[3][1] is None and abs(a[3][0] + a[3][2]) < 1e-9


def test_pack_mask_aligns_with_next_token_targets(monkeypatch):
    import torch
    monkeypatch.setattr(torch.Tensor, "cuda", lambda self: self)
    ids, mask = pack([([5, 6, 7], [8, 1]), ([5], [9])])
    assert ids.tolist() == [[5, 6, 7, 8, 1], [5, 9, 0, 0, 0]]
    # targets are ids[:, 1:]; mask marks the completion tokens 8, 1 and 9
    assert mask.tolist() == [[0, 0, 1, 1], [1, 0, 0, 0]]
