from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List

import fitz  # PyMuPDF


@dataclass(frozen=True)
class PdfPage:
    page_number: int  # 1-based
    text: str


def extract_pdf_pages(pdf_path: Path) -> List[PdfPage]:
    doc = fitz.open(str(pdf_path))
    pages: List[PdfPage] = []
    for i in range(doc.page_count):
        page = doc.load_page(i)
        text = page.get_text("text") or ""
        pages.append(PdfPage(page_number=i + 1, text=text))
    return pages


def pdf_title_from_path(pdf_path: Path) -> str:
    return pdf_path.stem
