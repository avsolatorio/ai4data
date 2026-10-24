"""Check a WebVTT caption file against subtitle guidelines, or rewrap it.

Reads a WebVTT file and reports, for each cue:

- a line longer than the line limit (default 37 characters);
- more lines than the line limit per cue (default 2);
- a reading speed above the limit in characters per second (default 20),
  counting every character of the cue text except line breaks;
- an end time that is not after the start time, or a start time before
  the start of the previous cue, which the WebVTT format does not allow;
- a cue that starts before the previous cue ends, which shows two
  captions at once.

The defaults come from published guidelines. The BBC Subtitle Guidelines
(version 1.2.5, March 2026) limit a line to 37 characters where subtitles
are shown in Teletext and recommend at most two lines for landscape video.
The Netflix English (USA) Timed Text Style Guide sets 42 characters per
line, two lines, and up to 20 characters per second for adult programmes.
The script also prints the median rate in words per minute, which the BBC
guidelines recommend at 160 to 180 for prepared subtitles, as information.

With --rewrap OUT, the script writes a copy in which each cue is divided
at sentence ends and between words into cues that respect the line limits,
with the time of the original cue shared by character count. This turns a
file with one cue per narration step, as some video tools write it, into
captions that a player can show.

Exit status 1 on any problem (without --rewrap).

Usage:
    python check_captions.py captions.vtt
    python check_captions.py captions.vtt --max-chars 42
    python check_captions.py tool_output.vtt --rewrap captions.vtt

Standard library only.

What this does not check: whether the caption text matches what the voice
says, whether a line break falls at a natural point in the sentence, or
the position and styling of the cues. A person watches the video with the
captions on for that.
"""

from __future__ import annotations

import argparse
import re
import statistics
import sys
from dataclasses import dataclass
from pathlib import Path

TIMING = re.compile(
    r"^((?:\d+:)?\d{2}:\d{2}\.\d{3})\s+-->\s+((?:\d+:)?\d{2}:\d{2}\.\d{3})(.*)$"
)


@dataclass
class Cue:
    start: float
    end: float
    lines: list[str]


def seconds(stamp: str) -> float:
    parts = [float(p) for p in stamp.split(":")]
    while len(parts) < 3:
        parts.insert(0, 0.0)
    return parts[0] * 3600 + parts[1] * 60 + parts[2]


def stamp(t: float) -> str:
    ms = round(t * 1000)
    return f"{ms // 3600000:02d}:{ms // 60000 % 60:02d}:{ms // 1000 % 60:02d}.{ms % 1000:03d}"


def parse(text: str) -> tuple[bool, list[Cue]]:
    blocks = re.split(r"\n\s*\n", text.replace("\r\n", "\n").lstrip("﻿"))
    header = blocks[0].startswith("WEBVTT")
    cues = []
    for block in blocks[1:] if header else blocks:
        lines = block.strip("\n").split("\n")
        if not lines or lines[0].startswith(("NOTE", "STYLE", "REGION")):
            continue
        if not TIMING.match(lines[0]) and len(lines) > 1:
            lines = lines[1:]  # cue identifier
        m = TIMING.match(lines[0])
        if m:
            cues.append(Cue(seconds(m.group(1)), seconds(m.group(2)), lines[1:]))
    return header, cues


def wrap(text: str, width: int) -> list[str]:
    lines: list[str] = []
    for word in text.split():
        if lines and len(lines[-1]) + 1 + len(word) <= width:
            lines[-1] += " " + word
        else:
            lines.append(word)
    return lines


def split_sentence(sentence: str, width: int, height: int) -> list[list[str]]:
    """Divide a sentence into cues of at most `height` lines of `width`.

    Lines are filled word by word. When the last cue would hold one short
    line, the words of the last two cues are shared between them at the
    middle, so that a cue does not end the sentence with a few words.
    """
    lines = wrap(sentence, width)
    chunks = [lines[i : i + height] for i in range(0, len(lines), height)]
    if len(chunks) > 1 and len(chunks[-1]) == 1 and len(chunks[-1][0]) < width / 2:
        words = " ".join(chunks[-2] + chunks[-1]).split()
        half = len(" ".join(words)) / 2
        used = 0
        for cut, word in enumerate(words, 1):
            used += len(word) + 1
            if used >= half:
                break
        first, second = (
            wrap(" ".join(words[:cut]), width),
            wrap(" ".join(words[cut:]), width),
        )
        if len(first) <= height and len(second) <= height and second:
            chunks[-2:] = [first, second]
    return chunks


def rewrap(cues: list[Cue], width: int, height: int) -> list[Cue]:
    out = []
    for cue in cues:
        sentences = re.split(r"(?<=[.!?])\s+", " ".join(cue.lines))
        chunks = []
        for sentence in sentences:
            chunks += split_sentence(sentence, width, height)
        total = sum(len(" ".join(c)) for c in chunks) or 1
        t = cue.start
        for chunk in chunks:
            share = (cue.end - cue.start) * len(" ".join(chunk)) / total
            out.append(Cue(t, t + share, chunk))
            t += share
    return out


def write(cues: list[Cue], path: Path) -> None:
    body = ["WEBVTT", ""]
    for cue in cues:
        body += [f"{stamp(cue.start)} --> {stamp(cue.end)}", *cue.lines, ""]
    path.write_text("\n".join(body), encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("captions", type=Path)
    parser.add_argument("--max-chars", type=int, default=37, help="characters per line")
    parser.add_argument("--max-lines", type=int, default=2, help="lines per cue")
    parser.add_argument(
        "--max-cps", type=float, default=20, help="characters per second"
    )
    parser.add_argument("--rewrap", type=Path, metavar="OUT")
    args = parser.parse_args(argv)

    header, cues = parse(args.captions.read_text(encoding="utf-8"))
    if args.rewrap:
        new = rewrap(cues, args.max_chars, args.max_lines)
        write(new, args.rewrap)
        print(f"wrote {args.rewrap.name}: {len(cues)} cues became {len(new)}")
        return 0

    problems: list[str] = []
    if not header:
        problems.append("the file does not start with WEBVTT")
    rates = []
    wpm = []
    for i, cue in enumerate(cues, 1):
        where = f"cue {i} ({stamp(cue.start)})"
        duration = cue.end - cue.start
        if duration <= 0:
            problems.append(
                f"{where}: ends at {stamp(cue.end)}, at or before its start"
            )
            continue
        if i > 1 and cue.start < cues[i - 2].start:
            problems.append(f"{where}: starts before the previous cue")
        elif i > 1 and cue.start < cues[i - 2].end:
            problems.append(
                f"{where}: starts before the previous cue ends at {stamp(cues[i - 2].end)}"
            )
        if len(cue.lines) > args.max_lines:
            problems.append(f"{where}: {len(cue.lines)} lines (limit {args.max_lines})")
        for line in cue.lines:
            if len(line) > args.max_chars:
                problems.append(
                    f"{where}: line of {len(line)} characters (limit {args.max_chars})"
                )
        chars = sum(len(line) for line in cue.lines) + len(cue.lines) - 1
        rate = chars / duration
        rates.append(rate)
        wpm.append(60 * len(" ".join(cue.lines).split()) / duration)
        if rate > args.max_cps:
            problems.append(
                f"{where}: {rate:.1f} characters per second (limit {args.max_cps:g})"
            )

    length = cues[-1].end if cues else 0
    print(f"{args.captions.name}: {len(cues)} cues, last cue ends at {stamp(length)}")
    if rates:
        print(
            f"reading speed: highest {max(rates):.1f} characters per second;"
            f" median {statistics.median(wpm):.0f} words per minute"
        )
    for p in problems:
        print(f"problem: {p}")
    print("result: " + ("FAIL" if problems else "PASS"))
    return 1 if problems else 0


if __name__ == "__main__":
    sys.exit(main())
