"""Record a short how-to video from a step script. See README.md before making a video.

A script is a JSON file with a title, a start URL, and a list of steps. Each step
names an action on the page (goto, click, highlight, scroll, card), a short
caption shown on screen, and a narration line read by Piper. The recorder
synthesises the narration first, so that each step stays on screen for as long
as its narration takes, then plays the steps in Chromium with Playwright while
recording the browser. ffmpeg combines the recording and the narration into an
MP4, and the narration lines are written to a WebVTT subtitle file.

A step may carry "speak", the text read aloud when it differs from the caption
in "say", for example numbers written out in words.

Voices: a Piper model name from voices/ (en_US-lessac-medium), or a Kokoro voice
(af_heart, am_michael, ...) when voices/kokoro holds the Kokoro model files.

Usage: .venv/bin/python record.py videos/document-collection.json [--voice NAME]
Output: out/<script name>.mp4 and out/<script name>.vtt
"""
import argparse
import json
import subprocess
import sys
import time
import wave
from pathlib import Path

from playwright.sync_api import sync_playwright

HERE = Path(__file__).resolve().parent
WIDTH, HEIGHT = 1280, 720
RATE = 24000

OVERLAY_JS = (HERE / "overlay.js").read_text()


_kokoro = None


def synthesise(text, voice, path, speed=1.0):
    """Writes the narration for one step as 16-bit mono at RATE; returns its length in seconds."""
    global _kokoro
    if (HERE / "voices" / f"{voice}.onnx").exists():
        raw = path.with_suffix(".piper.wav")
        subprocess.run(
            [str(HERE / ".venv/bin/piper"), "-m", str(HERE / "voices" / f"{voice}.onnx"),
             "--length-scale", f"{1.05 / speed:.3f}", "-f", str(raw)],
            input=text.encode(), check=True, capture_output=True)
        subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-i", str(raw), "-ar", str(RATE), "-ac", "1", str(path)], check=True)
    else:
        import soundfile
        from kokoro_onnx import Kokoro
        if _kokoro is None:
            _kokoro = Kokoro(str(HERE / "voices/kokoro/kokoro-v1.0.onnx"), str(HERE / "voices/kokoro/voices-v1.0.bin"))
        audio, sr = _kokoro.create(text, voice=voice, speed=speed, lang="en-us")
        assert sr == RATE
        soundfile.write(str(path), audio, sr, subtype="PCM_16")
    with wave.open(str(path)) as w:
        return w.getnframes() / w.getframerate()


def page_url(url):
    """A stage path such as stages/pcn.html becomes a file URL; web addresses pass through."""
    if "://" in url:
        return url
    return (HERE / url).resolve().as_uri()


def run_step(page, step, narrate):
    """Play one step. narrate() starts the narration and waits until it ends."""
    kind = step["do"]
    if kind == "goto":
        page.goto(page_url(step["url"]), wait_until="load")
        time.sleep(1.0)
    if kind == "eval":
        # Explainer stages: the step runs page code, such as go(scene, beat).
        page.evaluate(step["js"])
        time.sleep(step.get("settle", 0.4))
        narrate()
        return
    if kind == "card":
        page.evaluate("o => window.howto.card(o)", step)
        time.sleep(0.5)
        narrate()
        return
    page.evaluate("() => window.howto.card(null)")
    page.evaluate("() => window.howto.ring(null)")
    page.evaluate("o => window.howto.caption(o)", step.get("caption", ""))
    page.evaluate("on => window.howto.url(on)", bool(step.get("show_url")))
    if not step.get("spotlight"):
        page.evaluate("() => window.howto.spot(null)")
    if kind in ("goto", "wait"):
        narrate()
        return
    target = page.locator(step["target"]).first
    if kind == "scroll":
        target.evaluate("el => el.scrollIntoView({behavior: 'smooth', block: 'center'})")
        time.sleep(1.0)
    else:
        target.scroll_into_view_if_needed()
    if step.get("run"):
        target.evaluate(step["run"])
    if step.get("spotlight"):
        page.evaluate("s => window.howto.spot(s)", target.get_attribute("src"))
        time.sleep(0.4)
        target = page.locator("#howto-spot img")
    page.evaluate("b => window.howto.point(b)", target.bounding_box())
    time.sleep(0.8)
    if kind == "hover":
        target.hover()
        time.sleep(0.3)
    ring = page.locator(step["ring"]).first if step.get("ring") else target
    page.evaluate("b => window.howto.ring(b)", ring.bounding_box())
    if kind != "click":
        narrate()
        return
    # A click is narrated while the target is highlighted, then performed.
    narrate()
    page.evaluate("() => window.howto.press()")
    target.click()
    page.wait_for_load_state("load")
    time.sleep(0.9)
    page.evaluate("() => window.howto.ring(null)")
    if step.get("then_show"):
        page.locator(step["then_show"]).first.evaluate(
            "el => window.scrollTo({top: el.getBoundingClientRect().top + window.scrollY - 90, behavior: 'smooth'})")
        time.sleep(1.0)


def vtt_time(t):
    h, rem = divmod(t, 3600)
    m, s = divmod(rem, 60)
    return f"{int(h):02d}:{int(m):02d}:{s:06.3f}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("script")
    ap.add_argument("--voice", default="en_US-lessac-medium")
    ap.add_argument("--speed", type=float, default=1.0, help="speaking rate; 1.1 reads 10 percent faster")
    ap.add_argument("--music", action="store_true", help="add quiet background music from music.py")
    ap.add_argument("--music-level", type=float, default=0.3, help="music gain; 0.3 puts the music about 9 dB under the voice")
    args = ap.parse_args()

    spec = json.loads(Path(args.script).read_text())
    name = Path(args.script).stem
    work = HERE / "out" / name
    work.mkdir(parents=True, exist_ok=True)

    # Narration first: its length sets the time each step stays on screen.
    for i, step in enumerate(spec["steps"]):
        step["audio"], step["length"] = None, 0.0
        if step.get("say"):
            step["audio"] = work / f"step{i:02d}.wav"
            step["length"] = synthesise(step.get("speak", step["say"]), args.voice, step["audio"], args.speed)

    with sync_playwright() as p:
        browser = p.chromium.launch()
        context = browser.new_context(
            viewport={"width": WIDTH, "height": HEIGHT},
            record_video_dir=str(work), record_video_size={"width": WIDTH, "height": HEIGHT},
            color_scheme=spec.get("theme", "light"))
        context.add_init_script(OVERLAY_JS)
        page = context.new_page()
        start = time.monotonic()
        for step in spec["steps"]:
            def narrate(step=step):
                step["at"] = time.monotonic() - start
                # "min" holds a step for a set time, for example to match an animation.
                time.sleep(max(step["length"] + step.get("pause", 0.6), step.get("min", 0)))
            run_step(page, step, narrate)
        end = time.monotonic() - start
        video = Path(page.video.path())
        context.close()
        browser.close()

    # The first step loads the start page; the video begins once it has loaded.
    trim = max(spec["steps"][0]["at"] - 0.3, 0)
    frames = int((end - trim) * RATE) + RATE
    track = bytearray(frames * 2)
    for step in spec["steps"]:
        if not step["audio"]:
            continue
        with wave.open(str(step["audio"])) as w:
            data = w.readframes(w.getnframes())
        offset = int((step["at"] - trim) * RATE) * 2
        track[offset:offset + len(data)] = data
    audio = work / "narration.wav"
    with wave.open(str(audio), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(RATE)
        w.writeframes(bytes(track))

    out = HERE / "out" / f"{name}.mp4"
    inputs = ["-ss", f"{trim:.3f}", "-i", str(video), "-i", str(audio)]
    fmt = "aformat=sample_fmts=fltp:sample_rates=48000:channel_layouts=stereo"
    # Piper and Kokoro voices differ in level; loudnorm brings both to -16 LUFS.
    voice = f"[1:a]loudnorm=I=-16:TP=-1.5:LRA=11,aresample=48000,{fmt}"
    if args.music:
        # The music sits about 9 dB under the narration and drops a few dB while a line is read.
        music = work / "music.wav"
        subprocess.run([sys.executable, str(HERE / "music.py"), str(music), f"{end - trim:.3f}"], check=True)
        inputs += ["-i", str(music)]
        graph = (f"{voice},asplit=2[v1][v2];[2:a]{fmt},volume={args.music_level}[m];"
                 "[m][v1]sidechaincompress=threshold=0.05:ratio=2:attack=60:release=700[md];"
                 "[v2][md]amix=inputs=2:normalize=0[a]")
    else:
        graph = f"{voice}[a]"
    subprocess.run(
        ["ffmpeg", "-y", "-loglevel", "error", *inputs, "-filter_complex", graph,
         "-map", "0:v", "-map", "[a]", "-t", f"{end - trim:.3f}",
         "-c:v", "libx264", "-crf", "18", "-preset", "slow", "-pix_fmt", "yuv420p",
         "-r", "30", "-c:a", "aac", "-ar", "48000", "-b:a", "160k", "-movflags", "+faststart", str(out)],
        check=True)

    cues = ["WEBVTT", ""]
    for step in spec["steps"]:
        if step.get("say"):
            a = step["at"] - trim
            cues += [f"{vtt_time(a)} --> {vtt_time(a + step['length'])}", step["say"], ""]
    (HERE / "out" / f"{name}.vtt").write_text("\n".join(cues))
    print(f"{out} ({end - trim:.1f} s)")


if __name__ == "__main__":
    sys.exit(main())
