"""Extract the text of the downloaded reports, split it into chunks, and embed them.

    python build_store.py           # writes chunks.jsonl, store.json and prints the store's size

Text extraction uses PyMuPDF. Each PDF page is cleaned and split into
chunks of at most 800 characters at paragraph and sentence boundaries, with
80 characters of overlap. Cleaning removes running headers and footers (a
short block in the top or bottom margin that recurs on more than five pages
of a report), page numbers, and photo credits. A chunk never crosses a page, so every chunk
carries one page number. The build leaves out front matter (table of
contents, lists of figures and tables, abbreviations, acknowledgements),
reference lists, pages with fewer than 200 characters of text, and chunks
shorter than 80 characters. A heading is kept with the paragraph after
it.

Each chunk's metadata: id, series, country, title, edition, date (YYYY-MM),
page (1-based PDF page), chunk (index of the chunk within its page).
"""

from __future__ import annotations

import json
import re
import sys

import pymupdf
from common import (
    CHUNKS_JSONL,
    EMBEDDING_MODEL,
    PDF_DIR,
    STORE_JSON,
    WORK,
    make_embeddings,
    sources,
)
from langchain_core.documents import Document
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_text_splitters import RecursiveCharacterTextSplitter

FRONT_MATTER = re.compile(
    r"^\s*(TABLE OF CONTENTS|CONTENTS|LIST OF (FIGURES|TABLES|BOXES|ABBREVIATIONS|ACRONYMS)|ABBREVIATIONS|ACRONYMS|"
    r"ACKNOWLEDGE?MENTS?|ABBREVIATIONS AND ACRONYMS)\b",
    re.IGNORECASE | re.MULTILINE,
)
DROP_LINE = re.compile(
    r"^(Public Disclosure Authorized|Photo:.*|.*\|\s*Edition No\.?\s*\d+|"
    r"Uganda Economic Update\s*\|.*|.*\|\s*Uganda Economic Update.*)$",
    re.IGNORECASE,
)
CONTROL = re.compile(r"\u00ad\s*|[\x00-\x08\x0b-\x1f\x7f]")  # soft hyphens (with the space after) and control characters
PAGE_NUMBER = re.compile(r"^([ivxlc]+|\d{1,3})$", re.IGNORECASE)
MIN_PAGE_CHARS = 200
MIN_CHUNK_CHARS = 80
CHUNK_SIZE, CHUNK_OVERLAP = 800, 80


def header_key(text: str) -> str:
    """A block's text without a page number at either end ("UGANDA: ECONOMIC UPDATE 12")."""
    return re.sub(r"^[ivxlc\d]{1,4}\s+|\s+[ivxlc\d]{1,4}$", "", " ".join(text.split()), flags=re.IGNORECASE)


def running_headers(pdf) -> set[str]:
    """Running headers and footers: short blocks with letters, in the top or bottom margin, on more than five pages."""
    pages: dict[str, set[int]] = {}
    for i, page in enumerate(pdf):
        height = page.rect.height
        for block in page.get_text("blocks"):
            if 0.08 * height < block[1] and block[3] < 0.92 * height:
                continue  # headers and footers sit in the top or bottom 8 percent of the page
            text = header_key(block[4])
            if 0 < len(text) < 80 and re.search(r"[A-Za-z]", text):
                pages.setdefault(text, set()).add(i)
    return {t for t, p in pages.items() if len(p) > 5}


def join_lines(lines: list[str]) -> str:
    text = ""
    for ln in lines:
        if text.endswith("-") and not text.endswith(" -"):
            text += ln  # a word broken across lines keeps its hyphen ("private-sector")
        else:
            text += (" " if text else "") + ln
    return re.sub(r"\s+", " ", text)


def page_text(page, headers: set[str]) -> str:
    """The page as paragraphs.

    PyMuPDF returns text blocks in the order of the content stream, which
    follows the columns of these reports. Some reports put each line in its
    own block, so consecutive blocks are merged into one paragraph when they
    are in the same column and separated by less than half a line, or when a
    sentence continues from the foot of one column to the top of the next.
    Sorting the blocks by position would interleave the lines of two columns.
    """
    paras: list[list] = []  # [text, x0, y1, line height]
    for block in page.get_text("blocks"):
        if block[6] != 0:  # image block
            continue
        if header_key(block[4]) in headers:
            continue
        lines = [ln.strip() for ln in block[4].splitlines()]
        lines = [ln for ln in lines if ln and not DROP_LINE.match(ln)]
        if not lines or (len(lines) == 1 and PAGE_NUMBER.match(lines[0])):  # empty, or a page number
            continue
        x0, y0, _x1, y1 = block[:4]
        text = CONTROL.sub("", join_lines(lines))
        height = (y1 - y0) / len(lines)
        if paras:
            last = paras[-1]
            same_column = abs(x0 - last[1]) < 20 and -1 <= y0 - last[2] < 0.5 * last[3]
            continues = not re.search(r"[.:;!?)\]]$", last[0]) and text[:1].islower()
            if same_column or continues:
                last[0] = join_lines([last[0], text])
                last[2], last[3] = y1, height
                continue
        paras.append([text, x0, y1, height])
    # Runs of short fragments (axis labels and legends of charts) become one line.
    out: list[str] = []
    in_run = False
    for text, *_ in paras:
        short = len(text) < 25
        if short and in_run and len(out[-1]) < 200:
            out[-1] += " " + text
        else:
            out.append(text)
        in_run = short
    # A heading (a short paragraph without closing punctuation) stays with the paragraph after it.
    merged: list[str] = []
    for text in out:
        if merged and len(merged[-1]) < 120 and not re.search(r"[.:;!?]$", merged[-1]) and "\n" not in merged[-1]:
            merged[-1] += "\n" + text
        else:
            merged.append(text)
    return "\n\n".join(merged)


CITATION = re.compile(r"\b(19|20)\d\d[a-z]?\.\s|\((19|20)\d\d[a-z]?(, [^)]*)?\)\.")
SOURCE_MARK = re.compile(r"https?://|www\.|Washington, ?D\.?C|Retrieved from|Accessed")


def is_front_matter(text: str) -> bool:
    head = text[:300]
    return bool(FRONT_MATTER.search(head)) or text.count("....") + text.count(". . . .") > 5


def is_references(text: str) -> bool:
    """A page of a reference list: it opens with the heading, or it is dense with citations or links."""
    if re.search(r"\b(REFERENCES|References|BIBLIOGRAPHY|Bibliography)\b", text[:150]):
        return True
    cites = len(CITATION.findall(text))
    return (cites >= 6 and 1000 * cites / len(text) >= 4) or len(SOURCE_MARK.findall(text)) >= 10


def main() -> int:
    manifest = sources()
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP, separators=["\n\n", ". ", "; ", ", ", " ", ""]
    )
    chunks: list[Document] = []
    stats = []
    for src in manifest["documents"]:
        path = PDF_DIR / src["file"]
        if not path.exists():
            raise SystemExit(f"{path} is missing; run fetch.py first")
        pdf = pymupdf.open(path)
        headers = running_headers(pdf)
        kept_pages = 0
        meta = {k: src[k] for k in ("id", "series", "country", "title", "edition", "date")}
        for i, page in enumerate(pdf):
            text = page_text(page, headers)
            if len(text) < MIN_PAGE_CHARS or is_front_matter(text) or is_references(text):
                continue
            kept_pages += 1
            pieces = [p for p in splitter.split_text(text) if len(p) >= MIN_CHUNK_CHARS]
            for j, piece in enumerate(pieces):
                chunks.append(Document(page_content=piece, metadata={**meta, "page": i + 1, "chunk": j}))
        stats.append((src["id"], len(pdf), kept_pages))

    WORK.mkdir(parents=True, exist_ok=True)
    with CHUNKS_JSONL.open("w") as fh:
        for c in chunks:
            fh.write(json.dumps({"metadata": c.metadata, "text": c.page_content}, ensure_ascii=False) + "\n")

    embeddings = make_embeddings()
    store = InMemoryVectorStore(embeddings)
    store.add_documents(chunks, ids=[f"{c.metadata['id']}/p{c.metadata['page']}/{c.metadata['chunk']}" for c in chunks])
    store.dump(str(STORE_JSON))

    # Size of the store, in the embedding model's word pieces and in characters.
    tok = embeddings._client.tokenizer
    lengths = [len(tok(c.page_content, add_special_tokens=False)["input_ids"]) for c in chunks]
    max_len = embeddings._client.max_seq_length
    pages_total = sum(s[1] for s in stats)
    pages_kept = sum(s[2] for s in stats)
    summary = {
        "documents": len(stats),
        "pdf_pages": pages_total,
        "pages_with_chunks": pages_kept,
        "chunks": len(chunks),
        "characters": sum(len(c.page_content) for c in chunks),
        "tokens_embedding_model": sum(lengths),
        "embedding_model": EMBEDDING_MODEL,
        "embedding_max_tokens": max_len,
        "chunks_over_max_tokens": sum(1 for n in lengths if n > max_len - 2),
        "per_document": [{"id": s[0], "pdf_pages": s[1], "pages_with_chunks": s[2],
                          "chunks": sum(1 for c in chunks if c.metadata["id"] == s[0])} for s in stats],
    }
    (WORK / "store_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
