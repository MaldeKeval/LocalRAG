from __future__ import annotations

from pathlib import Path
from typing import List, Tuple

from docx import Document


def extract_docx_blocks(docx_path: Path) -> List[Tuple[int, str]]:
    """
    Extract DOCX content as synthetic numbered text blocks.

    DOCX does not have stable page numbers, so we use paragraph order
    as integer block indices to stay compatible with the existing chunker.
    """
    doc = Document(str(docx_path))
    blocks: List[Tuple[int, str]] = []
    block_idx = 1
    for para in doc.paragraphs:
        text = (para.text or "").strip()
        if not text:
            continue
        blocks.append((block_idx, text))
        block_idx += 1
    return blocks
