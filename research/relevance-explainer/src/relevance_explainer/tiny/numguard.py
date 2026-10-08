"""Number guard: the explanation may only write numbers that occur in the input.

Explanations copy years, counts, thresholds and percentages from the metadata and the search, and the 20M model
sometimes garbles them ("$1.000 a day" for $1.90; 0.6% of explanations on the realistic test, 1.9% on the nine-type
set). At each decoding step where a number is being written, tokens are allowed only if the number they form is a
prefix of a number in the input (decimal comma and point, and thousands separators, are treated as equivalent);
a number may only end where it is complete. Small integers (0-20) stay free for counts the model derives.
`repair_ranges` then swaps a written year range that runs backwards when both years occur in the input.
"""
from __future__ import annotations

import re

import torch

NUM = re.compile(r"\d+(?:[.,]\d+)*")
RUN = re.compile(r"\d[\d.,]*")
TAIL = re.compile(r"\d[\d.,]*$")
FREE = {str(i) for i in range(21)}
DIGITS = "0123456789"  # ASCII only: str.isdigit() is also true for superscripts such as "²"


def _forms(s: str) -> set[str]:
    out = {s, s.replace(",", ""), s.replace(".", ""), s.replace(",", "."), s.replace(".", ",")}
    for f in list(out):  # 1.90 -> 1.9 (and 1,90 -> 1,9)
        if re.fullmatch(r"\d+[.,]\d*0", f):
            out.add(f.rstrip("0"))
    return out


class NumberGuard:
    def __init__(self, tk):
        self.tk = tk
        n = tk.get_vocab_size()
        self.strs = [tk.decode([i]) for i in range(n)]
        self.digit_ids = [i for i, s in enumerate(self.strs) if any(c in DIGITS for c in s)]
        d = lambda ch: ch != "" and ch in DIGITS  # noqa: E731  ("" in DIGITS is True)
        self.cont_ids = [i for i, s in enumerate(self.strs) if d(s[:1]) or (s[:1] in ".," and d(s[1:2]))]
        self.cont_t = torch.tensor(self.cont_ids)

    def state(self, input_text: str) -> dict:
        full = set(FREE)
        for m in NUM.findall(input_text):
            full |= _forms(m)
        prefixes = {f[:k] for f in full for k in range(1, len(f) + 1)}
        return {"full": full, "prefixes": prefixes, "text": ""}

    @staticmethod
    def _runs_ok(st: dict, text: str, open_end: bool) -> bool:
        """Every number in `text` is complete and allowed, except a last one still open at the end, which must be a prefix."""
        runs = list(RUN.finditer(text))
        for k, m in enumerate(runs):
            r = m.group(0)
            if open_end and k == len(runs) - 1 and m.end() == len(text):
                if r not in st["prefixes"]:
                    return False
            elif r.rstrip(".,") not in st["full"]:
                return False
        return True

    def apply(self, st: dict, logits: torch.Tensor) -> None:
        """Mask, in place, tokens that would write a number absent from the input. `st["text"]` is the text so far."""
        tail_m = TAIL.search(st["text"][-40:])
        tail = tail_m.group(0) if tail_m else ""
        top = int(logits.argmax())
        if not tail and not any(c in DIGITS for c in self.strs[top]):
            return
        if tail:
            # a trailing "." or "," after a complete number is sentence punctuation unless a digit follows
            complete = tail.rstrip(".,") in st["full"]
            if not complete:  # the number must continue: only valid continuations
                keep = [i for i in self.cont_ids if self._runs_ok(st, tail + self.strs[i], True)]
                mask = torch.ones_like(logits, dtype=torch.bool)
                if keep:
                    mask[keep] = False
                    logits.masked_fill_(mask, float("-inf"))
                return
            bad = [i for i in self.cont_ids if not self._runs_ok(st, tail + self.strs[i], True)]
            bad += [i for i in self.digit_ids if self.strs[i][:1] not in DIGITS and self.strs[i][:1] not in ".,"
                    and not self._runs_ok(st, self.strs[i], True)]
        else:
            bad = [i for i in self.digit_ids if not self._runs_ok(st, self.strs[i], True)]
        if bad and len(bad) < logits.numel():
            logits[bad] = float("-inf")

    def advance(self, st: dict, token: int) -> None:
        st["text"] = (st["text"] + self.strs[token])[-200:]


RANGE = re.compile(r"(?<!\d)((?:19|20)\d\d)(\s*(?:-|–|—|to|a|à|y|et|and|до|по|hingga|sampai|ile|hadi|से|إلى|至|到)\s*)((?:19|20)\d\d)(?!\d)")


def repair_ranges(text: str, input_text: str) -> str:
    """Swap a year range written backwards ("2014-2007") when both years occur in the input."""
    have = set(re.findall(r"(?<!\d)(?:19|20)\d\d(?!\d)", input_text))

    def fix(m):
        a, sep, b = m.group(1), m.group(2), m.group(3)
        return f"{b}{sep}{a}" if int(b) < int(a) and a in have and b in have else m.group(0)
    return RANGE.sub(fix, text)


# ---------- sequence-level check: used instead of the token constraint ----------
# Measured 2026-10-08: the token constraint above changed 7 of 1,495 realistic-test explanations, mostly for the worse
# (when the model wants a number the input lacks, a forced alternative is another wrong number or garbled text), and its
# first version also broke "2010," into "2010,²". Resampling whole explanations keeps the text natural.
_SPACE_THOUSANDS = re.compile(r"(\d)[   ](\d{3})(?!\d)")
_SHORT_RANGE = re.compile(r"(?<!\d)((?:19|20)\d\d)\s*[-–/]\s*(\d\d)(?!\d)")


def _join_thousands(s: str) -> str:
    """"143 000" and "1 000" (space as thousands separator) as one number."""
    prev = None
    while prev != s:
        prev, s = s, _SPACE_THOUSANDS.sub(r"\1\2", s)
    return s


def supported_numbers(input_text: str) -> set[str]:
    full = set(FREE)
    text = _join_thousands(input_text)
    for m in NUM.findall(text):
        full |= _forms(m)
    for m in _SHORT_RANGE.finditer(text):  # "1960-85", "2019/20": the second year written out
        full.add(m.group(1)[:2] + m.group(2))
    return full


def unsupported(text: str, input_text: str) -> list[str]:
    """Numbers in the narrative (after the verdict line) that occur nowhere in the input."""
    full = supported_numbers(input_text)
    narrative = _join_thousands(text.split("\n", 1)[-1])
    return [r for r in (m.rstrip(".,") for m in NUM.findall(narrative)) if r not in full]
