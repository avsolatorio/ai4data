"""Contact sheets for reviewing an explainer before and after recording.

  stage: plays the eval steps of a script in Chromium without narration and
         screenshots the stage after each step, so that layout problems show
         up before a five-minute recording.
  video: takes a frame from the finished MP4 just before the end of each
         narration line (times from the .vtt), which is the state the viewer
         sees while the line is read.

Each sheet holds nine frames at half size. Read the sheets, fix the stage,
and repeat.

Usage: .venv/bin/python review.py stage videos/pcn-explainer.json [--wait 3]
       .venv/bin/python review.py video videos/pcn-explainer.json
Output: out/<name>/review-stage-N.png or out/<name>/review-video-N.png
"""
import argparse
import json
import re
import subprocess
import tempfile
import time
from pathlib import Path

from PIL import Image
from record import page_url

HERE = Path(__file__).resolve().parent


def sheets(frames, prefix):
    paths = []
    for g in range(0, len(frames), 9):
        sheet = Image.new("RGB", (1920, 1080), "white")
        for i, f in enumerate(frames[g:g + 9]):
            sheet.paste(Image.open(f).resize((640, 360)), ((i % 3) * 640, (i // 3) * 360))
        path = f"{prefix}-{g // 9}.png"
        sheet.save(path)
        paths.append(path)
    return paths


def stage(spec, work, tmp, wait):
    from playwright.sync_api import sync_playwright
    frames = []
    with sync_playwright() as p:
        browser = p.chromium.launch()
        page = browser.new_page(viewport={"width": 1280, "height": 720})
        errors = []
        page.on("pageerror", lambda e: errors.append(str(e)))
        for i, step in enumerate(spec["steps"]):
            if step["do"] == "goto":
                page.goto(page_url(step["url"]), wait_until="load")
                time.sleep(1.5)
                continue
            if step["do"] != "eval":
                continue
            page.evaluate(step["js"])
            # Long animations carry a "min"; the others settle within a few seconds.
            time.sleep(max(step.get("min", 0), wait))
            frames.append(f"{tmp}/s{i:03d}.png")
            page.screenshot(path=frames[-1])
        browser.close()
    if errors:
        print("page errors:", *errors, sep="\n  ")
    return sheets(frames, work / "review-stage")


def video(name, work, tmp):
    vtt = (HERE / "out" / f"{name}.vtt").read_text()
    ends = re.findall(r"--> (\d\d):(\d\d):(\d\d\.\d+)", vtt)
    mp4 = HERE / "out" / f"{name}.mp4"
    # The last cue can end at the last frame; a seek there returns no frame.
    duration = float(subprocess.run(["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "csv=p=0", str(mp4)],
                                    capture_output=True, text=True, check=True).stdout)
    frames = []
    for i, (h, m, s) in enumerate(ends):
        t = min(int(h) * 3600 + int(m) * 60 + float(s) - 0.2, duration - 0.5)
        frames.append(f"{tmp}/v{i:03d}.png")
        subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-ss", f"{t:.2f}", "-i", str(mp4),
                        "-frames:v", "1", frames[-1]], check=True)
    return sheets(frames, work / "review-video")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["stage", "video"])
    ap.add_argument("script")
    ap.add_argument("--wait", type=float, default=3.0, help="seconds to wait after each step in stage mode")
    args = ap.parse_args()
    spec = json.loads(Path(args.script).read_text())
    name = Path(args.script).stem
    work = HERE / "out" / name
    work.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory() as tmp:
        paths = stage(spec, work, tmp, args.wait) if args.mode == "stage" else video(name, work, tmp)
    print(*paths, sep="\n")


if __name__ == "__main__":
    main()
