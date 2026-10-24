"""Light, upbeat background music, synthesised here so that no licence applies.

112 beats per minute in C major, one chord per bar over I-V-vi-IV (C, G, Am, F).
A soft kick on beats 1 and 3, a closed hi-hat on the off-beats, a bass line on the
chord roots, plucked chords on a syncopated pattern, and a quiet pad underneath.
The result is a 48 kHz stereo WAV of the requested length with a fade-in and a
fade-out. record.py mixes it under the narration.

Usage: .venv/bin/python music.py OUT.wav SECONDS [--seed N]
"""
import argparse
import wave

import numpy as np

SR = 48000
BPM = 112
BEAT = 60 / BPM
BAR = 4 * BEAT
# chord tones (MIDI) and bass root for C, G, Am, F
CHORDS = [([60, 64, 67], 36), ([59, 62, 67], 43), ([60, 64, 69], 45), ([60, 65, 69], 41)]


def hz(note):
    return 440.0 * 2 ** ((note - 69) / 12)


def place(mix, sig, at, gain, pan=0.5):
    s = int(at * SR)
    if s >= mix.shape[1]:
        return
    e = min(mix.shape[1], s + len(sig))
    mix[0, s:e] += sig[:e - s] * gain * (1 - pan) * 2
    mix[1, s:e] += sig[:e - s] * gain * pan * 2


def kick():
    t = np.arange(int(0.35 * SR)) / SR
    f = 50 + 70 * np.exp(-t * 30)
    return np.sin(2 * np.pi * np.cumsum(f) / SR) * np.exp(-t * 11)


def hat(rng):
    n = int(0.05 * SR)
    noise = rng.standard_normal(n)
    noise = np.diff(noise, prepend=0)  # high-pass
    return noise * np.exp(-np.arange(n) / SR * 90)


def pluck(note, seconds=0.45):
    t = np.arange(int(seconds * SR)) / SR
    f = hz(note)
    v = np.sin(2 * np.pi * f * t) + 0.35 * np.sin(4 * np.pi * f * t) + 0.12 * np.sin(6 * np.pi * f * t)
    return v * np.exp(-t * 7) * np.minimum(1, t * 400)


def bass(note, seconds):
    t = np.arange(int(seconds * SR)) / SR
    f = hz(note)
    v = np.sin(2 * np.pi * f * t) + 0.25 * np.sin(4 * np.pi * f * t)
    return v * np.minimum(1, t * 200) * np.exp(-t * 2.5)


def pad(notes, seconds):
    t = np.arange(int(seconds * SR)) / SR
    v = sum(np.sin(2 * np.pi * hz(n) * t) + np.sin(2 * np.pi * (hz(n) + 0.4) * t) for n in notes)
    env = np.minimum(1, t / 0.4) * np.minimum(1, (seconds - t) / 0.4)
    return v * env / (2 * len(notes))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("out")
    ap.add_argument("seconds", type=float)
    ap.add_argument("--seed", type=int, default=7)
    args = ap.parse_args()
    rng = np.random.default_rng(args.seed)

    n = int(args.seconds * SR)
    mix = np.zeros((2, n + SR))
    k, h = kick(), hat(rng)
    bar = 0
    while bar * BAR < args.seconds:
        t0 = bar * BAR
        notes, root = CHORDS[bar % 4]
        place(mix, pad(notes, BAR), t0, 0.10)
        for b in range(4):
            if b in (0, 2):
                place(mix, k, t0 + b * BEAT, 0.55)
            place(mix, h, t0 + (b + 0.5) * BEAT, 0.10, pan=0.65)
        # bass: root on 1, 2-and, 3, and the fifth on 4
        for at, nt, ln in ((0, root, 0.6), (1.5, root, 0.4), (2, root, 0.6), (3, root + 7, 0.5)):
            place(mix, bass(nt, ln * BEAT * 2), t0 + at * BEAT, 0.32)
        # plucked chord: 1, 1-and-a, 2-and, 3-and, 4
        for at in (0, 0.75, 1.5, 2.5, 3):
            for j, nt in enumerate(notes):
                place(mix, pluck(nt + 12), t0 + at * BEAT + j * 0.012, 0.07, pan=0.35 + 0.15 * j)
        # a short melody every other bar
        if bar % 2 == 1:
            for at, nt in ((0, notes[2] + 12), (0.5, notes[1] + 12), (1, notes[0] + 12), (2.5, notes[1] + 12)):
                place(mix, pluck(nt + 12, 0.6), t0 + at * BEAT, 0.05, pan=0.6)
        bar += 1

    mix = mix[:, :n]
    fade_in, fade_out = int(1.5 * SR), int(3.0 * SR)
    mix[:, :fade_in] *= np.linspace(0, 1, fade_in)
    mix[:, n - fade_out:] *= np.linspace(1, 0, fade_out)
    mix /= np.abs(mix).max() + 1e-9
    data = (mix.T * 0.9 * 32767).astype("<i2").tobytes()
    with wave.open(args.out, "wb") as w:
        w.setnchannels(2)
        w.setsampwidth(2)
        w.setframerate(SR)
        w.writeframes(data)


if __name__ == "__main__":
    main()
