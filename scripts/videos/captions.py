"""Divide narration lines into WebVTT cues that follow subtitle guidelines.

record.py calls cues() to write the subtitle file of a video. Each narration
line is divided at sentence ends and between words into cues of at most two
lines of 37 characters, the limits of the BBC Subtitle Guidelines that the
explainer videos cookbook checks with check_captions.py. The time of the
narration line, with the pause after it, is shared between its cues by
character count; the pause keeps the reading speed of fast lines under 20
characters per second.

Run as a script, it rewraps a subtitle file that record.py wrote before this
division existed, with one cue per narration line ending when the voice stops:

    python captions.py out/pcn-explainer.vtt out/pcn-explainer.vtt

Each line is then held for the pause after it, or until the next line starts.

Standard library only.
"""
import re
import sys
from pathlib import Path

WIDTH, HEIGHT = 37, 2
HOLD = 0.6  # the default pause of record.py after each narration line
TIMING = re.compile(r"^((?:\d+:)?\d{2}:\d{2}\.\d{3})\s+-->\s+((?:\d+:)?\d{2}:\d{2}\.\d{3})")


def stamp(t):
    ms = round(t * 1000)
    return f"{ms // 3600000:02d}:{ms // 60000 % 60:02d}:{ms // 1000 % 60:02d}.{ms % 1000:03d}"


def seconds(text):
    parts = [float(p) for p in text.split(":")]
    while len(parts) < 3:
        parts.insert(0, 0.0)
    return parts[0] * 3600 + parts[1] * 60 + parts[2]


def wrap(text, width=WIDTH):
    lines = []
    for word in text.split():
        if lines and len(lines[-1]) + 1 + len(word) <= width:
            lines[-1] += " " + word
        else:
            lines.append(word)
    return lines


def split_sentence(sentence, width=WIDTH, height=HEIGHT):
    """Fill lines word by word; a last cue of one short line shares words with the cue before it."""
    lines = wrap(sentence, width)
    chunks = [lines[i:i + height] for i in range(0, len(lines), height)]
    if len(chunks) > 1 and len(chunks[-1]) == 1 and len(chunks[-1][0]) < width / 2:
        words = " ".join(chunks[-2] + chunks[-1]).split()
        half = len(" ".join(words)) / 2
        used = 0
        for cut, word in enumerate(words, 1):
            used += len(word) + 1
            if used >= half:
                break
        first, second = wrap(" ".join(words[:cut]), width), wrap(" ".join(words[cut:]), width)
        if len(first) <= height and len(second) <= height and second:
            chunks[-2:] = [first, second]
    return chunks


def cues(start, end, text):
    """Return (start, end, lines) cues for one narration line read from start to end."""
    chunks = []
    for sentence in re.split(r"(?<=[.!?])\s+", text.strip()):
        chunks += split_sentence(sentence)
    total = sum(len(" ".join(c)) for c in chunks) or 1
    out, t = [], start
    for chunk in chunks:
        share = (end - start) * len(" ".join(chunk)) / total
        out.append((t, t + share, chunk))
        t += share
    return out


def vtt(items):
    """Write WebVTT text from (start, end, lines) cues."""
    body = ["WEBVTT", ""]
    for a, b, lines in items:
        body += [f"{stamp(a)} --> {stamp(b)}", *lines, ""]
    return "\n".join(body)


def rewrap(path, out):
    lines = []
    for block in re.split(r"\n\s*\n", Path(path).read_text(encoding="utf-8"))[1:]:
        rows = block.strip("\n").split("\n")
        m = TIMING.match(rows[0]) if rows else None
        if m:
            lines.append([seconds(m.group(1)), seconds(m.group(2)), " ".join(rows[1:])])
    for this, after in zip(lines, lines[1:] + [None]):
        this[1] = min(this[1] + HOLD, after[0]) if after else this[1] + HOLD
    items = []
    for a, b, text in lines:
        items += cues(a, b, text)
    Path(out).write_text(vtt(items), encoding="utf-8")
    print(f"{out}: {len(items)} cues")


if __name__ == "__main__":
    rewrap(*sys.argv[1:3])
