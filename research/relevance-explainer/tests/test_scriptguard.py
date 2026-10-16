from pathlib import Path

import pytest
import torch
from tokenizers import Tokenizer

from relevance_explainer.tiny.model import Config, TinyLM
from relevance_explainer.tiny.scriptguard import banned_for, classes_of, token_masks

TOK = Path(__file__).resolve().parent / "fixtures" / "tokenizer-12k.json"  # the first-round tokenizer (data/corpus/tokenizer.json)


def test_classes_of_lead_bytes():
    assert classes_of("Привет".encode()) == {"cyrillic"}
    assert classes_of("عتبر".encode()) == {"arabic"}
    assert classes_of("नमस्ते".encode()) == {"indic"}
    assert classes_of("贫困".encode()) == {"cjk"}
    assert classes_of("café — “x”".encode()) == set()  # Latin-1, dashes and quotes are not blocked


@pytest.mark.skipif(not TOK.exists(), reason="tokenizer not built")
def test_banned_for_language_and_input():
    tk = Tokenizer.from_file(str(TOK))
    masks = token_masks(tk)
    ar = tk.encode("عتبر").ids[0]
    ru = tk.encode("Ð").ids  # noqa: F841 (not a real word; only checks encode works)
    latin = tk.encode(" lugar").ids[0]
    es = banned_for(masks, "Respond in: Spanish\nQuery: datos")
    assert es[ar] and not es[latin]
    assert not banned_for(masks, "Respond in: Arabic\nQuery: بيانات")[ar]
    # a script that occurs in the input (an Arabic title) is allowed even for a Spanish answer
    assert not banned_for(masks, "Respond in: Spanish\nResult: تقرير")[ar]


def test_generate_batch_respects_banned():
    torch.manual_seed(0)
    m = TinyLM(Config(vocab=50, d_model=32, n_layer=2, n_head=2, d_ff=64, max_len=64)).eval()
    free = m.generate_batch([[3, 4, 5]], eos=1, max_new=8, no_repeat=0)[0]
    ban = torch.zeros(1, 50, dtype=torch.bool)
    ban[0, list(set(free))] = True
    out = m.generate_batch([[3, 4, 5]], eos=1, max_new=8, no_repeat=0, banned=ban)[0]
    assert not set(out) & set(free)
