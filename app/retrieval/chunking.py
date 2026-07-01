"""
Chunking: turn raw policy documents into retrievable chunks.

Primary path is semantic-aware chunking: split into sentences, then pack
sentences into windows that respect a soft token budget with overlap. Also
loads .txt/.md and .pdf (via pdfplumber when available).
"""
from __future__ import annotations

import logging
import re
from pathlib import Path
from uuid import uuid4

from app.models.schemas import RetrievedChunk

logger = logging.getLogger(__name__)


def _split_sentences(text: str) -> list[str]:
    # Lightweight sentence splitter; good enough for policy prose. For higher
    # quality, swap in spaCy's sentence segmentation (doc.sents).
    text = re.sub(r"\s+", " ", text.strip())
    parts = re.split(r"(?<=[.!?])\s+", text)
    return [p.strip() for p in parts if p.strip()]


def chunk_text(
    text: str,
    doc_id: str,
    target_tokens: int = 180,
    overlap_sentences: int = 1,
) -> list[RetrievedChunk]:
    """
    Pack sentences into chunks up to ~target_tokens (whitespace token estimate),
    with a sentence-level overlap so answers spanning a boundary aren't lost.
    """
    sentences = _split_sentences(text)
    if not sentences:
        return []

    chunks: list[RetrievedChunk] = []
    current: list[str] = []
    current_tokens = 0

    def flush(sents: list[str]) -> None:
        if sents:
            chunks.append(
                RetrievedChunk(
                    chunk_id=uuid4().hex,
                    doc_id=doc_id,
                    text=" ".join(sents),
                )
            )

    for sent in sentences:
        n = len(sent.split())
        if current and current_tokens + n > target_tokens:
            flush(current)
            # start next chunk with an overlap tail
            current = current[-overlap_sentences:] if overlap_sentences else []
            current_tokens = sum(len(s.split()) for s in current)
        current.append(sent)
        current_tokens += n

    flush(current)
    return chunks


def _read_pdf(path: Path) -> str:
    try:
        import pdfplumber

        text_parts: list[str] = []
        with pdfplumber.open(str(path)) as pdf:
            for page in pdf.pages:
                text_parts.append(page.extract_text() or "")
        return "\n".join(text_parts)
    except Exception as exc:  # pragma: no cover - optional dep
        logger.warning("Could not read PDF %s (%s); skipping.", path, exc)
        return ""


def load_and_chunk_dir(policies_dir: str) -> list[RetrievedChunk]:
    """Load every .txt/.md/.pdf policy file in a directory and chunk it."""
    chunks: list[RetrievedChunk] = []
    root = Path(policies_dir)
    if not root.exists():
        return chunks
    for path in sorted(root.glob("**/*")):
        suffix = path.suffix.lower()
        if suffix in {".txt", ".md"}:
            text = path.read_text(encoding="utf-8", errors="ignore")
        elif suffix == ".pdf":
            text = _read_pdf(path)
        else:
            continue
        chunks.extend(chunk_text(text, doc_id=path.name))
    return chunks
