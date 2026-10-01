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


def test_copy_bias_restricts_later_tokens_to_the_prompt_and_spares_the_start():
    torch.manual_seed(3)
    m = _small()
    prompt = [5, 6, 7, 8, 9, 10]
    plain = m.generate_batch([prompt], eos=-1, max_new=12, no_repeat=0)[0]
    biased = m.generate_batch([prompt], eos=-1, max_new=12, no_repeat=0, copy_bias=1e4, copy_skip=4)[0]
    assert biased[:4] == plain[:4]  # the verdict region is untouched
    assert all(t in prompt for t in biased[4:])


def test_int8_export_round_trip_matches_fake_quantization(tmp_path):
    from relevance_explainer.tiny.export import export_int8, load_int8

    torch.manual_seed(4)
    m = _small()
    tk_json = '{"version":"1.0","truncation":null,"padding":null,"added_tokens":[],"normalizer":null,"pre_tokenizer":null,' \
              '"post_processor":null,"decoder":null,"model":{"type":"BPE","vocab":{"a":0},"merges":[]}}'
    export_int8(m.state_dict(), m.c.to_dict(), tk_json, tmp_path / "m.pt")
    loaded, _ = load_int8(tmp_path / "m.pt")
    q8, _ = quantize(m, 8)
    ids = torch.randint(2, 100, (1, 16))
    assert torch.allclose(loaded(ids), q8(ids), atol=1e-3)


def test_sampling_is_reproducible_with_a_seed_and_differs_from_greedy():
    m = _small()
    prompts = [torch.randint(2, 100, (n,)).tolist() for n in (6, 9)]
    torch.manual_seed(7)
    a = m.generate_batch(prompts, eos=-1, max_new=15, no_repeat=0, temperature=1.0, top_k=20)
    torch.manual_seed(7)
    b = m.generate_batch(prompts, eos=-1, max_new=15, no_repeat=0, temperature=1.0, top_k=20)
    greedy = m.generate_batch(prompts, eos=-1, max_new=15, no_repeat=0)
    assert a == b and a != greedy


def test_quantization_sizes_and_error():
    m = _small()
    q8, s8 = quantize(m, 8)
    q4, s4 = quantize(m, 4)
    assert s4 < s8 < m.n_params() * 4
    ids = torch.randint(2, 100, (1, 16))
    err8 = (m(ids) - q8(ids)).abs().mean().item()
    err4 = (m(ids) - q4(ids)).abs().mean().item()
    assert err8 < err4
