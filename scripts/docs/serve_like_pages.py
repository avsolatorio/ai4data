"""Serve website/build with the address lookup of GitHub Pages, for testing redirects.

GitHub Pages answers /x with x.html when that file exists, redirects /x to /x/ when
x is a folder, and answers /x/ with x/index.html. `docusaurus serve` and
`python3 -m http.server` follow other rules, so a redirect page that works on them
can loop on GitHub Pages. This server follows the GitHub Pages rules.

Usage: python3 scripts/docs/serve_like_pages.py [--port 8765]
The site is then at http://127.0.0.1:8765/ai4data/.
"""
import argparse
import http.server
import os
import tempfile
from pathlib import Path

BUILD = Path(__file__).resolve().parents[2] / "website" / "build"


def handler(root):
    class Handler(http.server.SimpleHTTPRequestHandler):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, directory=root, **kwargs)

        def send_head(self):
            path = self.path.split("?")[0].split("#")[0]
            target = os.path.join(root, path.lstrip("/"))
            if not path.endswith("/") and not os.path.splitext(path)[1]:
                if os.path.isfile(target + ".html"):
                    self.path = path + ".html"
                elif os.path.isdir(target):
                    self.send_response(301)
                    self.send_header("Location", path + "/")
                    self.end_headers()
                    return None
            return super().send_head()

    return Handler


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=8765)
    args = ap.parse_args()
    root = tempfile.mkdtemp()
    os.symlink(BUILD, os.path.join(root, "ai4data"))
    print(f"http://127.0.0.1:{args.port}/ai4data/")
    http.server.ThreadingHTTPServer(("127.0.0.1", args.port), handler(root)).serve_forever()


if __name__ == "__main__":
    main()
