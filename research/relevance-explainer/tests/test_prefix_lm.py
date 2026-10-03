import pytest
import torch

from relevance_explainer.tiny.model import Config, TinyLM


def _model(seed: int = 0, copy: bool = False) -> TinyLM:
    torch.manual_seed(seed)
    return TinyLM(Config(vocab=100, d_model=32, n_layer=2, n_head=2, d_ff=64, max_len=128, dropout=0.0,
                         prefix_lm=True, copy=copy, copy_dim=16)).eval()


def test_prompt_is_bidirectional_and_output_is_causal():
    m = _model()
    ids = torch.randint(2, 100, (1, 10))
    p = torch.tensor([6])
    base = m(ids, prefix=p)
    later_prompt = ids.clone()
    later_prompt[0, 5] = (later_prompt[0, 5] + 1) % 98 + 2
    # changing a prompt token changes the outputs at earlier prompt positions
    assert not torch.allclose(m(later_prompt, prefix=p)[0, 2], base[0, 2])
    later_out = ids.clone()
    later_out[0, 8] = (later_out[0, 8] + 1) % 98 + 2
    # changing an output token leaves every earlier position unchanged
    assert torch.allclose(m(later_out, prefix=p)[0, :8], base[0, :8], atol=1e-5)


def test_labels_give_the_prefix_and_missing_prefix_raises():
    m = _model(1)
    ids = torch.randint(2, 100, (2, 9))
    lab = torch.full_like(ids, -100)
    lab[0, 4:], lab[1, 6:] = ids[0, 4:], ids[1, 6:]
    logits, _ = m(ids, lab)
    assert torch.allclose(logits, m(ids, prefix=torch.tensor([4, 6])), atol=1e-5)
    with pytest.raises(ValueError):
        m(ids)


@pytest.mark.parametrize("copy", [False, True])
def test_batched_cached_decoding_matches_single_sequence(copy):
    m = _model(2, copy)
    prompts = [torch.randint(2, 100, (n,)).tolist() for n in (5, 12, 3)]
    for no_repeat in (0, 3):
        batched = m.generate_batch(prompts, eos=-1, max_new=12, no_repeat=no_repeat)
        for p, out in zip(prompts, batched):
            assert out == m.generate(torch.tensor([p]), eos=-1, max_new=12, no_repeat=no_repeat)
