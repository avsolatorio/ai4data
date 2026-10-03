import torch

from relevance_explainer.tiny.model import Config, TinyLM


def _copy_model(seed: int = 0) -> TinyLM:
    torch.manual_seed(seed)
    m = TinyLM(Config(vocab=100, d_model=32, n_layer=2, n_head=2, d_ff=64, max_len=128, dropout=0.0, copy=True, copy_dim=16)).eval()
    torch.nn.init.normal_(m.copy_gate.weight, std=0.5)  # make the gate input-dependent for the test
    return m


def test_copy_distribution_is_normalized_and_trainable():
    m = _copy_model()
    ids = torch.randint(2, 100, (3, 10))
    logp = m(ids)
    assert torch.allclose(logp.exp().sum(-1), torch.ones(3, 10), atol=1e-4)
    _, loss = m(ids, ids.clone())
    loss.backward()
    assert m.copy_q.weight.grad is not None and m.copy_gate.bias.grad is not None


def test_copy_head_can_only_point_backwards():
    m = _copy_model(1)
    ids = torch.randint(2, 100, (1, 8))
    full = m(ids)
    prefix = m(ids[:, :5])
    assert torch.allclose(full[:, :5], prefix, atol=1e-5)


def test_copy_batched_cached_decoding_matches_single_sequence():
    m = _copy_model(2)
    prompts = [torch.randint(2, 100, (n,)).tolist() for n in (5, 12, 3)]
    for no_repeat in (0, 3):
        batched = m.generate_batch(prompts, eos=-1, max_new=15, no_repeat=no_repeat)
        for p, out in zip(prompts, batched):
            assert out == m.generate(torch.tensor([p]), eos=-1, max_new=15, no_repeat=no_repeat)
