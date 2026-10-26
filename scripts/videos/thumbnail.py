"""Render a thumbnail page (thumbnails/<name>.html) to a 1280x720 PNG for YouTube.

The page is laid out at 1280x720 like a stage. YouTube accepts thumbnails
of 1280x720 up to 2 MB.

    .venv/bin/python thumbnail.py thumbnails/pcn.html
Output: out/<name>-thumbnail.png
"""
import sys
from pathlib import Path

from playwright.sync_api import sync_playwright

HERE = Path(__file__).resolve().parent


def main(page_path):
    page_path = Path(page_path).resolve()
    out = HERE / "out" / f"{page_path.stem}-thumbnail.png"
    with sync_playwright() as p:
        browser = p.chromium.launch()
        page = browser.new_page(viewport={"width": 1280, "height": 720})
        page.goto(page_path.as_uri(), wait_until="networkidle")
        page.wait_for_timeout(500)
        page.screenshot(path=str(out))
        browser.close()
    print(f"{out} ({out.stat().st_size / 1e6:.2f} MB)")


if __name__ == "__main__":
    sys.exit(main(sys.argv[1]))
