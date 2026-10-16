from pathlib import Path

import pytest
from tokenizers import Tokenizer

from relevance_explainer.tiny.train import batches, encode, from_verdict_last

TOK = Path(__file__).resolve().parent / "fixtures" / "tokenizer-12k.json"  # the first-round tokenizer (data/corpus/tokenizer.json)


def test_from_verdict_last():
    assert from_verdict_last("Covers Kenya.\nVerdict: relevant") == "relevant\nCovers Kenya."
    assert from_verdict_last("Two\nlines.\nVerdict: partly_relevant") == "partly_relevant\nTwo\nlines."
    assert from_verdict_last("no tag") == "invalid\nno tag"


@pytest.mark.skipif(not TOK.exists(), reason="tokenizer not built")
def test_encode_verdict_last_marks_the_verdict_tokens(monkeypatch):
    import torch
    monkeypatch.setattr(torch.Tensor, "cuda", lambda self: self)
    tk = Tokenizer.from_file(str(TOK))
    row = {"input": "Query: x", "output": "partly_relevant\nThe item covers Kenya only.", "verdict": "partly_relevant"}
    (e,), _ = encode(tk, [row], 512, verdict_last=True)
    target = tk.decode(e["ids"][e["plen"]:])
    assert target.startswith("The item covers Kenya only.") and target.rstrip().endswith("Verdict: partly_relevant")
    assert tk.decode(e["ids"][e["vstart"] : e["vstart"] + e["vlen"]]).strip() == "partly_relevant"
    ids, lab, wt, _ = next(batches([e], 1, False, None, vw=5.0))
    assert wt[0, e["vstart"]] == 5.0 and wt[0, e["plen"]] == 1.0
    (f,), _ = encode(tk, [row], 512)
    assert f["vstart"] == f["plen"]
    ids, lab, wt, _ = next(batches([f], 1, False, None, vw=5.0, pw=0.1))
    p = f["plen"]
    assert (lab[0, 1:p] == ids[0, 1:p]).all() and lab[0, 0] == -100
    assert torch.allclose(wt[0, :p], torch.full((p,), 0.1)) and wt[0, p] == 5.0
