"""Number guard: numbers in the explanation must occur in the input; backward year ranges are repaired."""
import torch
from tokenizers import Tokenizer

from relevance_explainer.tiny.numguard import NumberGuard, repair_ranges

TK = Tokenizer.from_file("data/corpus/tokenizer.json")
NG = NumberGuard(TK)
INPUT = "Result: Poverty headcount ratio at $1.90 a day\nYears: 1981-2019; Coverage: 167 countries"


def greedy_number(prefix: str, prefer: str) -> str:
    """Write a number after `prefix` by always preferring the tokens of `prefer` (a garbled number); the guard decides."""
    st = NG.state(INPUT)
    for t in TK.encode(prefix).ids:
        NG.advance(st, t)
    out = ""
    want = TK.encode(prefer).ids
    for k in range(8):
        logits = torch.zeros(TK.get_vocab_size())
        if k < len(want):
            logits[want[k]] = 10.0
        for i in NG.cont_ids[:400]:
            logits[i] += 0.1
        sp = TK.encode(" a").ids[0]
        logits[sp] += 5.0
        NG.apply(st, logits)
        nxt = int(logits.argmax())
        NG.advance(st, nxt)
        out += NG.strs[nxt]
        if not any(c.isdigit() or c in ".," for c in NG.strs[nxt]):
            break
    return out


def test_numbers_in_input_pass():
    st = NG.state(INPUT)
    for s in ("1.90", "1,90", "1981", "2019", "167", "3"):
        assert NG._runs_ok(st, s, False), s


def test_invented_number_is_blocked():
    st = NG.state(INPUT)
    assert not NG._runs_ok(st, "1.000", False)
    assert not NG._runs_ok(st, "2025", False)
    written = greedy_number("living below $", "1.000")
    assert "1.000" not in written


def test_punctuation_after_a_number_is_free():
    """'2019,' then a word: the comma is punctuation, not the start of a decimal (no forced continuation)."""
    st = NG.state(INPUT)
    for t in TK.encode("covers 1981 to 2019,").ids:
        NG.advance(st, t)
    logits = torch.zeros(TK.get_vocab_size())
    word = TK.encode(" and").ids[0]
    logits[word] = 5.0
    NG.apply(st, logits)
    assert int(logits.argmax()) == word


def test_no_superscripts_or_empty_tokens_as_digits():
    assert all(any(c in "0123456789" for c in NG.strs[i]) for i in NG.digit_ids)
    assert all(NG.strs[i][:1] in "0123456789.," and NG.strs[i] for i in NG.cont_ids)


def test_repair_ranges():
    assert repair_ranges("covers 2019-1981 annually", INPUT) == "covers 1981-2019 annually"
    assert repair_ranges("from 2019 to 1981", INPUT) == "from 1981 to 2019"
    assert repair_ranges("covers 2019-1975", INPUT) == "covers 2019-1975"  # 1975 not in the input: left alone
