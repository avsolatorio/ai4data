"""Check that each cookbook follows the authoring standard.

For every folder under cookbook/ (except _template and authoring) the
checker verifies:

errors (exit status 1)
  - index.mdx exists with title and description, and has the required
    sections (Chapters and questions, Maturity levels, Files, Citation,
    Version history)
  - each chapter (a file that uses the Recipe component) has front matter
    (title "N. ...", sidebar_position, description), a **Question:** line,
    the required sections (Rationale, Target state, Maturity levels,
    Recipes, Common mistakes, Checklist), and a Checklist component
  - each Recipe has a title "N.M ...", a level from the shared vocabulary,
    skills, time, and a **Result:** line
  - headings and titles do not use the banned constructions (", not ",
    "rather than", a statement plus a comma and a tag)
  - the cookbook is registered in website/sidebars-cookbook.js and
    website/src/content/cookbooks.js
  - template placeholders ({{...}}), guidance comments, and sample text
    are gone

warnings
  - standards.md, glossary.md, or contributing.md missing
  - a recipe without a "What this does not do" note
  - second person (you, your) in prose outside code blocks
  - Implementation options or Verification section missing
  - no folder under website/static/cookbook-files/<id>/

Usage:
    python scripts/docs/check_cookbooks.py
    python scripts/docs/check_cookbooks.py --id ai-ready-dissemination
    python scripts/docs/check_cookbooks.py --strict   # warnings fail too
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
COOKBOOK = REPO / "cookbook"
SIDEBARS = REPO / "website" / "sidebars-cookbook.js"
CARDS = REPO / "website" / "src" / "content" / "cookbooks.js"
STATIC = REPO / "website" / "static" / "cookbook-files"
SKIP = {"_template", "authoring"}

LEVELS = {"Foundational", "AI-ready", "AI-native"}
INDEX_SECTIONS = ["Chapters and questions", "Maturity levels", "Files", "Citation", "Version history"]
CHAPTER_REQUIRED = ["Rationale", "Target state", "Maturity levels", "Recipes", "Common mistakes", "Checklist"]
CHAPTER_RECOMMENDED = ["Implementation options", "Verification"]

BANNED = re.compile(r", not |rather than|instead of|not just|isn.t\.|\bmerely\b|\bsimply\b|\bactually\b")
COMMA_TAG = re.compile(r", (and|with|but|so|then|because) ")
SECOND_PERSON = re.compile(r"\b(you|your|yours)\b", re.IGNORECASE)
FRONT = re.compile(r"^---\n(.*?)\n---\n", re.DOTALL)
RECIPE = re.compile(r"<Recipe\s+([^>]*)>(.*?)</Recipe>", re.DOTALL)
ATTR = re.compile(r'(\w+)="([^"]*)"')
PLACEHOLDER = re.compile(r"\{\{[A-Z_]+\}\}")
GUIDANCE = re.compile(r"\{/\*.*?\*/\}|<!--.*?-->", re.DOTALL)  # template guidance comments
TEMPLATE_TEXT = (
    "Imperative title naming the result",
    "one sentence on what exists when the recipe is done",
    "First item",
)


class Report:
    def __init__(self) -> None:
        self.errors: list[str] = []
        self.warnings: list[str] = []

    def error(self, where: Path, msg: str) -> None:
        self.errors.append(f"{where.relative_to(REPO)}: {msg}")

    def warn(self, where: Path, msg: str) -> None:
        self.warnings.append(f"{where.relative_to(REPO)}: {msg}")


def front_matter(text: str) -> dict[str, str]:
    m = FRONT.match(text)
    if not m:
        return {}
    out = {}
    for line in m.group(1).splitlines():
        if ":" in line:
            k, v = line.split(":", 1)
            out[k.strip()] = v.strip().strip('"').strip("'")
    return out


def headings(text: str) -> list[str]:
    return [m.group(2).strip() for m in re.finditer(r"^(#{1,4}) (.+)$", text, re.MULTILINE)]


def prose(text: str) -> str:
    """Text without fenced code blocks, front matter, and JSX attribute values."""
    text = FRONT.sub("", text)
    text = re.sub(r"```.*?```", "", text, flags=re.DOTALL)
    text = re.sub(r"<[A-Z]\w*[^>]*>", "", text)
    return text


def is_serial_list(title: str) -> bool:
    """'A, B, and C' is a list; 'A thing, with a tag' is not."""
    m = COMMA_TAG.search(title)
    if not m:
        return True
    before = title[: m.start()]
    return "," in before  # a second comma earlier means a serial list


def check_titles(path: Path, text: str, rep: Report) -> None:
    fm = front_matter(text)
    titles = (
        headings(text)
        + [fm.get("title", "")]
        + [m.group(2) for m in re.finditer(r'(title|label)="([^"]*)"', text)]
    )
    for t in titles:
        if not t:
            continue
        if BANNED.search(t):
            rep.error(path, f"banned construction in title: {t!r}")
        if COMMA_TAG.search(t) and not is_serial_list(t):
            rep.error(path, f"comma-and-tag title: {t!r}")
    body = prose(text)
    for m in BANNED.finditer(body):
        line = body.count("\n", 0, m.start()) + 1
        rep.error(path, f"banned construction near line {line}: {m.group(0)!r}")
    if SECOND_PERSON.search(body):
        rep.warn(path, "second person (you/your) in prose; use 'the organization' or a passive form")


def check_index(path: Path, rep: Report) -> None:
    text = path.read_text(encoding="utf-8")
    fm = front_matter(text)
    for key in ("title", "description"):
        if not fm.get(key):
            rep.error(path, f"front matter lacks {key}")
    hs = headings(text)
    for section in INDEX_SECTIONS:
        if section not in hs:
            rep.error(path, f"missing section '## {section}'")
    if PLACEHOLDER.search(text):
        rep.error(path, "template placeholders remain")
    check_unfinished(path, text, rep)
    check_titles(path, text, rep)


def check_unfinished(path: Path, text: str, rep: Report) -> None:
    """Template guidance comments and sample text mean the page is unfinished."""
    if GUIDANCE.search(text):
        rep.error(path, "template guidance comments remain; replace them with content")
    body = prose(text)  # the recipe template is allowed inside a code block
    for sample in TEMPLATE_TEXT:
        if sample in body:
            rep.error(path, f"template sample text remains: {sample!r}")


def check_chapter(path: Path, rep: Report) -> None:
    text = path.read_text(encoding="utf-8")
    fm = front_matter(text)
    if not re.match(r"^\d+\. ", fm.get("title", "")):
        rep.error(path, 'front matter title must read "N. Title"')
    for key in ("sidebar_position", "description"):
        if not fm.get(key):
            rep.error(path, f"front matter lacks {key}")
    if "**Question:**" not in text:
        rep.error(path, "missing '**Question:** ...' line under the title")
    hs = headings(text)
    for section in CHAPTER_REQUIRED:
        if section not in hs:
            rep.error(path, f"missing section '## {section}'")
    for section in CHAPTER_RECOMMENDED:
        if section not in hs and not any(h.startswith(section) for h in hs):
            rep.warn(path, f"recommended section '## {section}' missing")
    if "<Checklist" not in text:
        rep.error(path, "missing Checklist component")
    recipes = list(RECIPE.finditer(text))
    if not recipes:
        rep.error(path, "no Recipe component found")
    for m in recipes:
        attrs = dict(ATTR.findall(m.group(1)))
        title = attrs.get("title", "")
        if not re.match(r"^\d+\.\d+ ", title):
            rep.error(path, f'recipe title must read "N.M Title": {title!r}')
        if attrs.get("level") not in LEVELS:
            rep.error(path, f"recipe {title!r}: level must be one of {sorted(LEVELS)}")
        for key in ("skills", "time"):
            if not attrs.get(key):
                rep.error(path, f"recipe {title!r}: missing {key}")
        if "**Result:**" not in m.group(2):
            rep.error(path, f"recipe {title!r}: missing '**Result:**' line")
        if "**What this does not do" not in m.group(2) and "**What this does not check" not in m.group(2):
            rep.warn(path, f"recipe {title!r}: no 'What this does not do' note")
    if PLACEHOLDER.search(text):
        rep.error(path, "template placeholders remain")
    check_unfinished(path, text, rep)
    check_titles(path, text, rep)


def check_cookbook(folder: Path, rep: Report) -> None:
    index = folder / "index.mdx"
    if not index.exists():
        rep.error(folder, "index.mdx missing")
    else:
        check_index(index, rep)
    chapters = [
        p
        for p in sorted(folder.glob("*.mdx"))
        if p.name != "index.mdx" and "<Recipe" in p.read_text(encoding="utf-8")
    ]
    if not chapters:
        rep.error(folder, "no chapter files (files that use the Recipe component)")
    for p in chapters:
        check_chapter(p, rep)
    for p in sorted(folder.glob("*.md")) + [
        p for p in sorted(folder.glob("*.mdx")) if p not in chapters and p.name != "index.mdx"
    ]:
        text = p.read_text(encoding="utf-8")
        if PLACEHOLDER.search(text):
            rep.error(p, "template placeholders remain")
        check_unfinished(p, text, rep)
        check_titles(p, text, rep)
    for name in ("standards.md", "glossary.md"):
        if not (folder / name).exists():
            rep.warn(folder, f"{name} missing")
    if not (folder / "contributing.md").exists() and not (folder / "contribute.md").exists():
        rep.warn(folder, "contributing.md missing")
    cid = folder.name
    if f"dirName: '{cid}'" not in SIDEBARS.read_text():
        rep.error(SIDEBARS, f"no sidebar for '{cid}'")
    if f"id: '{cid}'" not in CARDS.read_text():
        rep.error(CARDS, f"no landing card for '{cid}'")
    if not (STATIC / cid).exists():
        rep.warn(folder, f"no folder website/static/cookbook-files/{cid}/")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--id", help="check one cookbook")
    parser.add_argument("--strict", action="store_true", help="treat warnings as errors")
    args = parser.parse_args(argv)

    folders = [p for p in sorted(COOKBOOK.iterdir()) if p.is_dir() and p.name not in SKIP]
    if args.id:
        folders = [p for p in folders if p.name == args.id]
        if not folders:
            sys.exit(f"no cookbook named {args.id!r}")

    rep = Report()
    for folder in folders:
        check_cookbook(folder, rep)

    for w in rep.warnings:
        print(f"warning  {w}")
    for e in rep.errors:
        print(f"error    {e}")
    print(f"\n{len(folders)} cookbook(s): {len(rep.errors)} error(s), {len(rep.warnings)} warning(s)")
    failed = bool(rep.errors) or (args.strict and bool(rep.warnings))
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
