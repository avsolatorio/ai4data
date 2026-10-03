"""Decoding constraint against stray characters from scripts that the answer should not contain.

The student sometimes emits fragments of another script inside a word (Arabic letters in a Spanish sentence, for
example). The tokenizer is byte-level BPE, so a token may hold whole characters or only some bytes of one. Each token
is classed by the UTF-8 lead bytes it contains, which are never continuation bytes:

  cyrillic  D0-D3 (U+0400-04FF)     arabic  D8-DB (U+0600-06FF)
  indic     E0    (U+0800-0FFF, Devanagari and the other Indic scripts)
  cjk       E4-E9 (U+4000-9FFF, CJK ideographs)

At each step a token is blocked when it carries a class that is neither the script of the requested language nor
present anywhere in the input. Latin, punctuation and other characters are never blocked. A continuation-only token
cannot start a character, so it is left alone.
"""
from __future__ import annotations

import re

import torch
from tokenizers import Tokenizer

LEADS = {"cyrillic": range(0xD0, 0xD4), "arabic": range(0xD8, 0xDC), "indic": range(0xE0, 0xE1), "cjk": range(0xE4, 0xEA)}
LANG_SCRIPT = {"Russian": "cyrillic", "Arabic": "arabic", "Hindi": "indic", "Chinese": "cjk"}
_LANG = re.compile(r"^Respond in:\s*(.+)$", re.M)


def _byte_decoder() -> dict[str, int]:
    """Inverse of the GPT-2 byte-to-unicode map used by byte-level BPE."""
    bs = list(range(ord("!"), ord("~") + 1)) + list(range(ord("¡"), ord("¬") + 1)) + list(range(ord("®"), ord("ÿ") + 1))
    cs = bs[:]
    n = 0
    for b in range(256):
        if b not in bs:
            bs.append(b)
            cs.append(256 + n)
            n += 1
    return {chr(c): b for b, c in zip(bs, cs)}


def classes_of(data: bytes) -> set[str]:
    return {name for name, r in LEADS.items() if any(b in r for b in data)}


def token_masks(tk: Tokenizer) -> dict[str, torch.Tensor]:
    dec = _byte_decoder()
    V = tk.get_vocab_size()
    masks = {name: torch.zeros(V, dtype=torch.bool) for name in LEADS}
    for i in range(V):
        tok = tk.id_to_token(i)
        if tok is None or any(ch not in dec for ch in tok):
            continue  # special tokens
        for name in classes_of(bytes(dec[ch] for ch in tok)):
            masks[name][i] = True
    return masks


def banned_for(masks: dict[str, torch.Tensor], text: str) -> torch.Tensor:
    """Tokens to block for one rendered input."""
    m = _LANG.search(text)
    allowed = classes_of(text.encode("utf-8")) | ({LANG_SCRIPT[m.group(1).strip()]} if m and m.group(1).strip() in LANG_SCRIPT else set())
    out = torch.zeros_like(next(iter(masks.values())))
    for name, mask in masks.items():
        if name not in allowed:
            out |= mask
    return out
