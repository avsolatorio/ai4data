"""Shared pieces of the public RAG example: paths, the store, citations, the prompt and the chain.

The retrieval stack and the prompt follow the session 01 notebooks
(training/session-01-building-ai-applications/notebooks/workshop.py and
03-retrieval-augmented-generation.ipynb): LangChain documents, the
sentence-transformers/all-MiniLM-L6-v2 embedding model, an
InMemoryVectorStore searched by cosine similarity, the same instructions,
and the same `make_rag` chain. Only the citation form differs: it names the
report edition and the PDF page, because these documents have pages and no
Markdown sections.
"""

from __future__ import annotations

import json
import os
import re
from pathlib import Path
from typing import Any, Callable

from langchain_core.documents import Document
from langchain_core.vectorstores import InMemoryVectorStore

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[3]
SOURCES = HERE / "sources.json"
WORK = Path(os.environ.get("RAG_PUBLIC_DIR", REPO / "local" / "rag-public"))
PDF_DIR = WORK / "pdf"
CHUNKS_JSONL = WORK / "chunks.jsonl"
STORE_JSON = WORK / "store.json"
RUNS_JSON = HERE / "retrieval_runs.json"
ANSWERS_JSON = HERE / "answers.json"

EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

# The instructions of notebook 03, word for word.
INSTRUCTIONS = """You are a research assistant for development data.
Answer only from the supplied passages. After each figure, cite the passage in the square-bracket form given.
If the passages do not contain what is needed, say so."""

HUMAN_TEMPLATE = "PASSAGES:\n{context}\n\nQUESTION: {question}"


def sources() -> dict[str, Any]:
    return json.loads(SOURCES.read_text())


def make_embeddings():
    """The embedding model of the session notebooks."""
    from langchain_huggingface import HuggingFaceEmbeddings

    try:
        import transformers.utils.logging as hf_logging

        hf_logging.disable_progress_bar()
    except Exception:
        pass
    return HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)


def load_store(embeddings=None) -> InMemoryVectorStore:
    if not STORE_JSON.exists():
        raise SystemExit(f"{STORE_JSON} not found; run fetch.py and build_store.py first")
    return InMemoryVectorStore.load(str(STORE_JSON), embeddings or make_embeddings())


def citation(doc: Document) -> str:
    """[keu-2026-07 · p. 12]: the report (series and edition month) and the PDF page."""
    m = doc.metadata
    return f"[{m.get('id', '?')} · p. {m.get('page', '?')}]"


def format_context(docs: list[Document]) -> str:
    """Each passage after its citation and the report's title, as in workshop.format_context."""
    return "\n\n".join(f"{citation(d)} {d.metadata.get('title', '')}\n{d.page_content}" for d in docs)


def since(date: str) -> Callable[[Document], bool]:
    """A metadata filter: keep chunks of reports dated `date` (YYYY-MM) or later."""
    return lambda d: str(d.metadata.get("date", "")) >= date


def country_is(country: str) -> Callable[[Document], bool]:
    return lambda d: d.metadata.get("country") == country


def both(*filters: Callable[[Document], bool]) -> Callable[[Document], bool]:
    return lambda d: all(f(d) for f in filters)


def hits(store: InMemoryVectorStore, question: str, k: int = 4, where=None) -> list[tuple[Document, float]]:
    """The search alone: (chunk, score) pairs, best first. The score is cosine similarity."""
    return store.similarity_search_with_score(question, k=k, filter=where)


def make_prompt():
    from langchain_core.prompts import ChatPromptTemplate

    return ChatPromptTemplate.from_messages([("system", INSTRUCTIONS), ("human", HUMAN_TEMPLATE)])


def make_rag(store: InMemoryVectorStore, model, k: int = 4, where=None):
    """The chain of notebook 03: search, passages in the prompt, the model."""
    from langchain_core.runnables import RunnableLambda, RunnablePassthrough

    retrieve = RunnableLambda(lambda q: store.similarity_search(q, k=k, filter=where))
    return {"context": retrieve | format_context, "question": RunnablePassthrough()} | make_prompt() | model


# Checks of the search, as in workshop.retrieval_hit and workshop.numbers_present.
NUMBER_RE = re.compile(r"(?<![\w.])-?(?:\d{1,3}(?:,\d{3})+|\d+)(?:\.\d+)?(?![\w.])")


def retrieval_hit(docs: list[Document], expected_docs: list[str]) -> bool:
    got = {d.metadata.get("id") for d in docs}
    return bool(got & set(expected_docs)) if expected_docs else True


def numbers_present(text: str, expected: list[dict[str, Any]], tol: float = 0.06) -> tuple[int, int]:
    nums = [float(n.replace(",", "")) for n in NUMBER_RE.findall(text)]
    found = sum(1 for e in expected if any(abs(n - e["value"]) <= max(tol, abs(e["value"]) * 0.005) for n in nums))
    return found, len(expected)


def figure_present(text: str, value: float) -> bool:
    """A stricter check than numbers_present: the value written as a percentage ("24 percent", "51.5%").

    numbers_present accepts any number within the tolerance, so in report text a
    short value such as 24 also matches "2023/24" or a figure for another year.
    """
    v = f"{value:g}"
    pattern = rf"(?<![\w.,/–-]){re.escape(v)}(?:\.0)?(?![\d])\s?(?:percent|per cent|%)"
    return re.search(pattern, " ".join(text.split())) is not None

