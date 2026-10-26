"""Render report pages as images for the explainer videos on this collection.

The videos show the real reports: a collage of the covers and inside pages,
and the page that a citation leads to, with the cited sentence marked. This
script renders those pages from the PDFs that fetch.py downloaded and writes,
next to them, pages.json (and the same data as pages.js for the stages) with the size of each image and the position of
each marked phrase on its page, as fractions of the page width and height.

The images are copies of the reports and stay out of git, with the PDFs, in
local/rag-public/pages/ (or $RAG_PUBLIC_DIR/pages/). The stages load them
from there.

    python render_pages.py

Requires PyMuPDF (requirements.txt).
"""

from __future__ import annotations

import json

import pymupdf
from common import CHUNKS_JSONL, PDF_DIR, STORE_JSON, WORK, make_embeddings

OUT = WORK / "pages"

# Inside pages for the collage: a text page, a table and a chart from
# several editions. Page numbers are PDF pages, as in the citations.
COLLAGE = [
    ("keu-2026-07", 12), ("ueu-2026-06", 20), ("keu-2024-12", 51),
    ("ueu-2025-09", 13), ("keu-2025-05", 14), ("ueu-2023-12", 17),
    ("ueu-2025-12", 37), ("keu-2023-06", 14),
]

# Pages that a citation in the videos leads to, with the phrases to mark.
CITED = {
    ("ueu-2025-12", 12): ["accelerated to 6.3 percent", "averages. Real gross domestic product (GDP) growth",
                          "rose to 51.3 percent of GDP, with domestic debt now"],
    ("ueu-2026-06", 20): ["3.0 percent"],
    ("ueu-2026-06", 22): ["USD 3.00-per-day line", "24 percent"],
    ("ueu-2026-06", 23): ["51.5 percent"],
    ("ueu-2023-12", 17): ["2.6 percent"],
    ("ueu-2023-06", 40): ["Real GDP growth (baseline)", "FY25f", "Table 4:", "Sources: UBOS, MoFPED, BoU"],
    ("ueu-2025-09", 21): ["grew by 6.8 percent"],
}

# Passages and questions whose vectors the videos draw: the first numbers
# of each vector, as stored (passages) or embedded here (questions).
VECTOR_PASSAGES = [f"ueu-2025-12/p12/{i}" for i in range(6)]
VECTOR_QUESTIONS = ["How much did Uganda's economy grow in FY25?"]
VECTOR_DIMS = 24

COVER_WIDTH, PAGE_WIDTH, CITED_WIDTH = 300, 520, 1600


def render(page, width, path):
    zoom = width / page.rect.width
    pix = page.get_pixmap(matrix=pymupdf.Matrix(zoom, zoom), alpha=False)
    pix.save(path)
    return {"file": path.name, "width": pix.width, "height": pix.height}


def passage_starts(page, rid, page_no):
    """Where each passage of the store begins on the page, found by its first words."""
    w, h = page.rect.width, page.rect.height
    out = []
    with CHUNKS_JSONL.open(encoding="utf-8") as f:
        for line in f:
            c = json.loads(line)
            m = c["metadata"]
            if m["id"] != rid or m["page"] != page_no:
                continue
            first = c["text"].lstrip(" .,;:\n").split()
            box = None
            for n in (7, 5, 3):
                hits = page.search_for(" ".join(first[:n])) if len(first) >= n else []
                if hits:
                    box = hits[0]
                    break
            out.append({"chunk": m["chunk"], "chars": len(c["text"]),
                        "start": [round(box.x0 / w, 4), round(box.y0 / h, 4), round(box.y1 / h, 4)] if box else None})
    return out


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    index = {"covers": [], "collage": [], "cited": []}
    for pdf in sorted(PDF_DIR.glob("*.pdf")):
        with pymupdf.open(pdf) as doc:
            item = render(doc[0], COVER_WIDTH, OUT / f"{pdf.stem}-cover.png")
            index["covers"].append({"id": pdf.stem, "pages": doc.page_count, **item})
    for rid, page_no in COLLAGE:
        with pymupdf.open(PDF_DIR / f"{rid}.pdf") as doc:
            item = render(doc[page_no - 1], PAGE_WIDTH, OUT / f"{rid}-p{page_no}.png")
            index["collage"].append({"id": rid, "page": page_no, **item})
    for (rid, page_no), phrases in CITED.items():
        with pymupdf.open(PDF_DIR / f"{rid}.pdf") as doc:
            page = doc[page_no - 1]
            item = render(page, CITED_WIDTH, OUT / f"{rid}-p{page_no}-large.png")
            w, h = page.rect.width, page.rect.height
            marks = []
            for phrase in phrases:
                hits = page.search_for(phrase)
                marks.append({"phrase": phrase, "boxes": [
                    [round(r.x0 / w, 4), round(r.y0 / h, 4), round(r.x1 / w, 4), round(r.y1 / h, 4)] for r in hits]})
            index["cited"].append({"id": rid, "page": page_no, "marks": marks,
                                   "passages": passage_starts(page, rid, page_no), **item})
    store = json.loads(STORE_JSON.read_text(encoding="utf-8"))
    index["vectors"] = {
        "dims": VECTOR_DIMS, "of": len(store[VECTOR_PASSAGES[0]]["vector"]),
        "passages": {k: [round(x, 4) for x in store[k]["vector"][:VECTOR_DIMS]] for k in VECTOR_PASSAGES},
        "questions": {q: [round(x, 4) for x in v[:VECTOR_DIMS]]
                      for q, v in zip(VECTOR_QUESTIONS, make_embeddings().embed_documents(VECTOR_QUESTIONS), strict=True)},
    }
    (OUT / "pages.json").write_text(json.dumps(index, indent=2) + "\n", encoding="utf-8")
    # The stages open from file:// and cannot fetch JSON, so they load this script.
    (OUT / "pages.js").write_text(f"window.PAGES = {json.dumps(index)};\n", encoding="utf-8")
    print(f"{OUT}: {len(index['covers'])} covers, {len(index['collage'])} pages, {len(index['cited'])} cited pages")
    for c in index["cited"]:
        print(" ", c["id"], c["page"], [(m["phrase"], len(m["boxes"])) for m in c["marks"]])


if __name__ == "__main__":
    main()
