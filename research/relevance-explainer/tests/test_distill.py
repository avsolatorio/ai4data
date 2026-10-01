from relevance_explainer.distill.onpolicy import quality
from relevance_explainer.distill.teacher_checks import class_of, strip_similarity


def test_class_of_handles_the_split_tokens_of_each_verdict():
    assert class_of("part") == "partly_relevant"
    assert class_of(" partly") == "partly_relevant"
    assert class_of("ir") == "irrelevant"
    assert class_of("irrelevant") == "irrelevant"
    assert class_of("relevant") == "relevant"
    assert class_of("related") is None
    assert class_of("not") is None


def test_strip_similarity_removes_only_that_line():
    text = "Query: q\nFilters: none\nRetrieval similarity: 0.61\nResult: r"
    assert strip_similarity(text) == "Query: q\nFilters: none\nResult: r"


def test_dpo_loss_is_log2_at_the_reference_and_falls_when_chosen_gains():
    import torch

    from relevance_explainer.tiny.dpo import dpo_loss

    z = torch.zeros(4)
    loss0, acc0 = dpo_loss(z, z, z, z, beta=0.1)
    assert abs(loss0.item() - 0.6931) < 1e-3 and acc0.item() == 0.0
    loss1, acc1 = dpo_loss(z + 5, z - 5, z, z, beta=0.1)
    assert loss1.item() < loss0.item() and acc1.item() == 1.0


def test_quality_prefers_faithful_and_requires_consistency():
    good = {"faithful": 5, "coherent": 5, "consistent": True, "language_ok": True}
    fluent_but_wrong = {"faithful": 2, "coherent": 5, "consistent": True, "language_ok": True}
    inconsistent = {"faithful": 5, "coherent": 5, "consistent": False, "language_ok": True}
    assert quality(good) > quality(fluent_but_wrong) > quality(inconsistent)
