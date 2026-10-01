import torch

from relevance_explainer.tiny.model import Config, TinyLM
from relevance_explainer.tiny.train import quantize


def _small():
    return TinyLM(Config(vocab=100, d_model=32, n_layer=2, n_head=4, d_ff=64, max_len=64, dropout=0.0)).eval()


def test_default_config_is_about_20m_parameters():
    n = TinyLM(Config()).n_params()
    assert 19e6 < n < 22e6


def test_forward_loss_is_finite_and_ignores_masked_labels():
    m = _small()
    ids = torch.randint(2, 100, (2, 16))
    labels = ids.clone()
    labels[:, :8] = -100
    logits, loss = m(ids, labels)
    assert logits.shape == (2, 16, 100) and torch.isfinite(loss)


def test_attention_is_causal():
    m = _small()
    a = torch.randint(2, 100, (1, 20))
    b = a.clone()
    b[0, 15:] = 5
    assert torch.allclose(m(a)[0, :15], m(b)[0, :15], atol=1e-5)


def test_generate_stops_at_eos_or_limit():
    m = _small()
    out = m.generate(torch.randint(2, 100, (1, 10)), eos=1, max_new=7)
    assert len(out) <= 7


def test_no_repeat_blocks_repeated_ngrams():
    torch.manual_seed(0)
    m = _small()
    # A tiny random model tends to loop; with blocking, no 3-gram may occur twice.
    out = m.generate(torch.randint(2, 100, (1, 8)), eos=-1, max_new=60, no_repeat=3)
    grams = [tuple(out[i : i + 3]) for i in range(len(out) - 2)]
    assert len(grams) == len(set(grams))


def test_batched_cached_decoding_matches_single_sequence():
    torch.manual_seed(1)
    m = _small()
    prompts = [torch.randint(2, 100, (n,)).tolist() for n in (5, 12, 9, 12, 3)]
    for no_repeat in (0, 3):
        batched = m.generate_batch(prompts, eos=-1, max_new=20, no_repeat=no_repeat)
        for p, out in zip(prompts, batched):
            single = m.generate(torch.tensor([p]), eos=-1, max_new=20, no_repeat=no_repeat)
            assert out == single


def test_batched_decoding_stops_each_sequence_at_eos():
    torch.manual_seed(2)
    m = _small()
    prompts = [torch.randint(2, 100, (n,)).tolist() for n in (4, 10)]
    first = m.generate_batch(prompts, eos=-1, max_new=12, no_repeat=2)  # blocks bigram loops so tokens vary
    # Use as EOS the first token of sequence 0 that has not been generated before, then expect a stop right there.
    j = next(i for i in range(1, 12) if first[0][i] not in first[0][:i])
    out = m.generate_batch(prompts, eos=first[0][j], max_new=12, no_repeat=2)
    assert out[0] == first[0][:j]
    assert len(out[1]) <= 12


def test_quantization_sizes_and_error():
    m = _small()
    q8, s8 = quantize(m, 8)
    q4, s4 = quantize(m, 4)
    assert s4 < s8 < m.n_params() * 4
    ids = torch.randint(2, 100, (1, 16))
    err8 = (m(ids) - q8(ids)).abs().mean().item()
    err4 = (m(ids) - q4(ids)).abs().mean().item()
    assert err8 < err4
