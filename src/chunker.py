"""Document chunking.

Implemented from scratch rather than via LangChain's text splitters so the
behaviour is fully transparent.  The strategy is paragraph-aware with a
character budget and a configurable overlap, which works well for Wikipedia
articles that are heavily paragraphed.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from config import CHUNK_OVERLAP_CHARS, CHUNK_SIZE_CHARS, MIN_CHUNK_CHARS


@dataclass
class Chunk:
    """A single chunk of text with the metadata needed for retrieval."""

    text: str
    source_title: str
    canonical_title: str
    url: str
    entity_type: str            # "person" | "place"
    chunk_index: int            # ordinal within the source document


# A loose sentence boundary -- Wikipedia plain text is well-formed enough
# that we can rely on punctuation followed by whitespace.
_SENTENCE_RE = re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9\"'])")


def _split_into_sentences(paragraph: str) -> list[str]:
    sentences = [s.strip() for s in _SENTENCE_RE.split(paragraph) if s.strip()]
    return sentences or [paragraph.strip()]


def _split_long_paragraph(paragraph: str, max_size: int) -> list[str]:
    """Break a paragraph that exceeds ``max_size`` along sentence boundaries."""
    sentences = _split_into_sentences(paragraph)
    out: list[str] = []
    buf = ""
    for sentence in sentences:
        if not buf:
            buf = sentence
        elif len(buf) + 1 + len(sentence) <= max_size:
            buf = f"{buf} {sentence}"
        else:
            out.append(buf)
            buf = sentence
            # If a single sentence is still oversized, hard-cut it.
            while len(buf) > max_size:
                out.append(buf[:max_size])
                buf = buf[max_size:]
    if buf:
        out.append(buf)
    return out


def chunk_text(
    text: str,
    *,
    source_title: str,
    canonical_title: str,
    url: str,
    entity_type: str,
    chunk_size: int = CHUNK_SIZE_CHARS,
    overlap: int = CHUNK_OVERLAP_CHARS,
) -> list[Chunk]:
    """Split ``text`` into overlapping chunks.

    Algorithm:

    1. Split on blank lines to get paragraphs.
    2. Greedily pack paragraphs into a buffer up to ``chunk_size`` characters.
    3. When emitting a chunk, seed the next buffer with the last ``overlap``
       characters of the current chunk, snapped to the previous whitespace
       boundary so words are never split.
    4. Paragraphs larger than ``chunk_size`` are pre-split along sentences.
    """

    if chunk_size <= overlap:
        raise ValueError("chunk_size must be greater than overlap")

    # Title gets prepended to the very first chunk so the embedding has a
    # strong signal of what the document is about even if the lede is short.
    paragraphs: list[str] = []
    title_block = f"{canonical_title}.\n\n" if canonical_title else ""
    if title_block:
        paragraphs.append(title_block.strip())
    for raw in text.split("\n\n"):
        para = raw.strip()
        if not para:
            continue
        if len(para) > chunk_size:
            paragraphs.extend(_split_long_paragraph(para, chunk_size))
        else:
            paragraphs.append(para)

    chunks: list[Chunk] = []
    buf = ""
    for para in paragraphs:
        candidate = para if not buf else f"{buf}\n\n{para}"
        if len(candidate) <= chunk_size:
            buf = candidate
            continue
        if buf:
            chunks.append(_emit(buf, source_title, canonical_title, url, entity_type, len(chunks)))
            buf = _carry_over(buf, overlap)
            buf = f"{buf}\n\n{para}" if buf else para
            # If even after carry-over the buffer overflows (e.g. paragraph
            # is gigantic), flush again.
            while len(buf) > chunk_size:
                chunks.append(_emit(buf[:chunk_size], source_title, canonical_title, url, entity_type, len(chunks)))
                buf = _carry_over(buf[:chunk_size], overlap) + buf[chunk_size:]
                if len(buf) <= chunk_size:
                    break
        else:
            # Paragraph itself longer than chunk_size even after sentence split
            # (very rare): hard-cut it.
            chunks.append(_emit(para[:chunk_size], source_title, canonical_title, url, entity_type, len(chunks)))
            buf = _carry_over(para[:chunk_size], overlap) + para[chunk_size:]

    if buf and len(buf) >= MIN_CHUNK_CHARS:
        chunks.append(_emit(buf, source_title, canonical_title, url, entity_type, len(chunks)))
    elif buf and chunks:
        # Append a tiny tail to the last chunk rather than wasting it.
        chunks[-1] = Chunk(
            text=f"{chunks[-1].text}\n\n{buf}".strip(),
            source_title=chunks[-1].source_title,
            canonical_title=chunks[-1].canonical_title,
            url=chunks[-1].url,
            entity_type=chunks[-1].entity_type,
            chunk_index=chunks[-1].chunk_index,
        )

    return chunks


def _emit(
    text: str,
    source_title: str,
    canonical_title: str,
    url: str,
    entity_type: str,
    index: int,
) -> Chunk:
    return Chunk(
        text=text.strip(),
        source_title=source_title,
        canonical_title=canonical_title,
        url=url,
        entity_type=entity_type,
        chunk_index=index,
    )


def _carry_over(buf: str, overlap: int) -> str:
    """Return the trailing ``overlap`` characters of ``buf``, snapped to a word boundary."""
    if overlap <= 0 or len(buf) <= overlap:
        return ""
    tail = buf[-overlap:]
    # Snap to the next whitespace so we don't start mid-word.
    space = tail.find(" ")
    if space == -1:
        return tail
    return tail[space + 1 :]
