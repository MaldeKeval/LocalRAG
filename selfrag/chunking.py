from __future__ import annotations

from dataclasses import dataclass
from typing import List, Sequence


@dataclass(frozen=True)
class TextChunk:
    text: str
    page_start: int
    page_end: int


def _clean_text(t: str) -> str:
    # Keep it conservative: specs have lots of formatting.
    t = t.replace("\u00ad", "")  # soft hyphen
    t = t.replace("\r\n", "\n").replace("\r", "\n")
    return "\n".join(line.rstrip() for line in t.split("\n")).strip()


def chunk_pages(
    pages: Sequence[tuple[int, str]],
    *,
    chunk_size: int,
    chunk_overlap: int,
    min_chunk_size: int,
) -> List[TextChunk]:
    """
    Chunk sequential page text while preserving page ranges.

    pages: list of (page_number, text)
    """
    assert chunk_size > 0
    assert 0 <= chunk_overlap < chunk_size

    cleaned = [(p, _clean_text(t)) for p, t in pages if _clean_text(t)]
    if not cleaned:
        return []

    chunks: List[TextChunk] = []
    current_parts: List[str] = []
    current_start = cleaned[0][0]
    current_end = cleaned[0][0]
    current_len = 0

    def flush(final: bool = False) -> None:
        nonlocal current_parts, current_len, current_start, current_end
        if not current_parts:
            return
        text = "\n\n".join(current_parts).strip()
        if len(text) >= min_chunk_size or final:
            chunks.append(TextChunk(text=text, page_start=current_start, page_end=current_end))

        # overlap by chars: keep tail of text in the next chunk buffer
        if chunk_overlap > 0 and text:
            tail = text[-chunk_overlap:]
            current_parts = [tail]
            current_len = len(tail)
            # page range for overlap is ambiguous; keep it as last page for safety
            current_start = current_end
        else:
            current_parts = []
            current_len = 0

    for page_num, text in cleaned:
        if not current_parts:
            current_start = page_num
            current_end = page_num
            current_parts = []
            current_len = 0

        # Add page text; flush if we'd exceed size
        addition = text.strip()
        add_len = len(addition) + (2 if current_parts else 0)
        if current_len + add_len > chunk_size and current_parts:
            flush()

        if not current_parts:
            current_start = page_num
        current_end = page_num
        current_parts.append(addition)
        current_len = sum(len(p) for p in current_parts) + (2 * (len(current_parts) - 1))

        if current_len >= chunk_size:
            flush()

    flush(final=True)
    return chunks
