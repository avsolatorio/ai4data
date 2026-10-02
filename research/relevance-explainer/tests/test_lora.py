import torch

from relevance_explainer.tiny.lora import add_lora, lora_bytes, lora_state, merge_lora
from relevance_explainer.tiny.model import Config, TinyLM


def _model():
    torch.manual_seed(0)
    return TinyLM(Config(vocab=100, d_model=32, n_layer=2, n_head=4, d_ff=64, max_len=64, dropout=0.0)).eval()


def test_adapter_starts_as_identity_and_only_adapter_params_train():
    m = _model()
    ids = torch.randint(2, 100, (2, 12))
    before = m(ids).clone()
    params = add_lora(m, rank=4, alpha=8.0)
    assert torch.allclose(m(ids), before, atol=1e-6)
    trainable = [p for p in m.parameters() if p.requires_grad]
    assert len(trainable) == len(params) == 2 * 5 * 2  # A and B for 5 targets in 2 blocks
    assert sum(p.numel() for p in trainable) < 0.2 * m.n_params()  # toy model; the real-size check is below


def test_merge_reproduces_adapted_outputs_and_removes_adapter_params():
    m = _model()
    add_lora(m, rank=4, alpha=8.0)
    with torch.no_grad():
        for k, p in m.named_parameters():
            if k.endswith(".B"):
                p.add_(torch.randn_like(p) * 0.1)
    ids = torch.randint(2, 100, (1, 10))
    adapted = m(ids).clone()
    merge_lora(m)
    assert torch.allclose(m(ids), adapted, atol=1e-5)
    assert not any(k.endswith(".A") or k.endswith(".B") for k in m.state_dict())


def test_lora_state_and_size():
    m = _model()
    add_lora(m, rank=4, alpha=8.0)
    st = lora_state(m)
    assert len(st) == 20
    assert lora_bytes(st, 8) == sum(v.numel() for v in st.values())


def test_real_size_adapter_is_about_one_megabyte():
    m = TinyLM(Config(vocab=12000, d_model=448, n_layer=7, n_head=7, d_ff=1024))
    add_lora(m, rank=16, alpha=32.0)
    assert 0.5e6 < lora_bytes(lora_state(m), 8) < 1.5e6
