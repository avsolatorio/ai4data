import torch

from relevance_explainer.tiny.model import Config, TinyLM
from relevance_explainer.viewer import filter_chips, parse_input, stream


def test_stream_matches_batched_greedy_decoding():
    for copy in (False, True):
        torch.manual_seed(0)
        m = TinyLM(Config(vocab=100, d_model=32, n_layer=2, n_head=2, d_ff=64, max_len=128, dropout=0.0, copy=copy, copy_dim=16)).eval()
        prompt = torch.randint(2, 100, (9,)).tolist()
        ref = m.generate_batch([prompt], eos=1, max_new=20, no_repeat=4)[0]
        got = [t for t, _ in stream(m, prompt, max_new=20, no_repeat=4)]
        got = got[:-1] if got and got[-1] == 1 else got
        assert got == ref


def test_parse_input_and_filter_chips():
    text = "Respond in: French\nQuery: x\nFilters: countries=Niger\nRetrieval similarity: 0.40\nResult: T\nUnit: %\nFilter check: country Niger: covered; years 2015: not covered"
    f = parse_input(text)
    assert f["Respond in"] == "French" and f["Retrieval similarity"] == "0.40" and ("Unit", "%") in f["meta"]
    assert [k for _, k in filter_chips(f["Filter check"])] == ["ok", "bad"]
