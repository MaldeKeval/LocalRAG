from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

from .chunking import TextChunk, chunk_pages
from .config import Settings
from .docx import extract_docx_blocks
from .pdf import extract_pdf_pages, pdf_title_from_path
from .util import load_json, manifest_key_for_pdf, normalize_path, save_json, sha256_file

MANIFEST_NAME = "manifest.json"


@dataclass(frozen=True)
class IngestDoc:
    doc_id: str
    source_path: str
    pdf_title: str
    file_hash: str
    chunk_id: str
    page_start: int
    page_end: int
    chunk_text: str
    created_at: str


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _migrate_manifest_key(key: str, pdf_root: Path) -> str:
    """Convert legacy absolute keys to paths relative to pdf_root."""
    k = key.strip().replace("\\", "/")
    p = Path(k)
    if not p.is_absolute():
        return k
    root = pdf_root.resolve()
    try:
        return p.resolve().relative_to(root).as_posix()
    except ValueError:
        pass
    try:
        return p.resolve(strict=False).relative_to(root).as_posix()
    except ValueError:
        pass
    return Path(k).name


def load_manifest(index_dir: Path, pdf_root: Path) -> Dict[str, Dict[str, str]]:
    """
    Manifest maps PDF path relative to the ingest root -> {file_hash, doc_id, updated_at}.
    Legacy manifests with absolute paths are rewritten on load.
    """
    raw = load_json(index_dir / MANIFEST_NAME, default={})
    if not raw:
        return {}
    out: Dict[str, Dict[str, str]] = {}
    for k, v in raw.items():
        nk = _migrate_manifest_key(k, pdf_root)
        out[nk] = v
    if out != raw:
        save_json(index_dir / MANIFEST_NAME, out)
    return out


def save_manifest(index_dir: Path, manifest: Dict[str, Dict[str, str]]) -> None:
    save_json(index_dir / MANIFEST_NAME, manifest)


SUPPORTED_EXTENSIONS = (".pdf", ".docx")


def iter_docs(pdf_dir: Path) -> Iterable[Path]:
    for ext in SUPPORTED_EXTENSIONS:
        for p in pdf_dir.rglob(f"*{ext}"):
            if p.is_file():
                yield p


def compute_doc_id(source_path: Path, file_hash: str) -> str:
    # Keep stable for identical content + path (path helps avoid collisions across duplicates).
    abs_path = normalize_path(source_path).encode("utf-8", errors="ignore")
    path_hash = __import__("hashlib").sha256(abs_path).hexdigest()
    return f"{file_hash[:16]}:{path_hash[:16]}"


def plan_ingest(settings: Settings, pdf_dir: Path) -> Tuple[List[Path], Dict[str, Dict[str, str]]]:
    root = pdf_dir.resolve()
    manifest = load_manifest(settings.index_dir, root)
    to_process: List[Path] = []
    for doc_path in iter_docs(pdf_dir):
        file_hash = sha256_file(doc_path)
        key = manifest_key_for_pdf(doc_path, root)
        prev = manifest.get(key)
        if not prev or prev.get("file_hash") != file_hash:
            to_process.append(doc_path)
    return to_process, manifest


def _extract_units(doc_path: Path) -> List[tuple[int, str]]:
    ext = doc_path.suffix.lower()
    if ext == ".pdf":
        pages = extract_pdf_pages(doc_path)
        return [(p.page_number, p.text) for p in pages]
    if ext == ".docx":
        return extract_docx_blocks(doc_path)
    raise ValueError(f"Unsupported file extension: {doc_path.suffix}")


def ingest_document(
    settings: Settings,
    doc_path: Path,
    *,
    file_hash: Optional[str] = None,
    pdf_root: Optional[Path] = None,
) -> List[IngestDoc]:
    file_hash = file_hash or sha256_file(doc_path)
    root = (pdf_root or settings.pdf_dir).resolve()
    source_path = manifest_key_for_pdf(doc_path, root)
    pdf_title = pdf_title_from_path(doc_path)
    doc_id = compute_doc_id(doc_path, file_hash)

    page_pairs = _extract_units(doc_path)
    chunks: List[TextChunk] = chunk_pages(
        page_pairs,
        chunk_size=settings.chunk_size,
        chunk_overlap=settings.chunk_overlap,
        min_chunk_size=settings.min_chunk_size,
    )

    created_at = _utc_now_iso()
    docs: List[IngestDoc] = []
    for idx, ch in enumerate(chunks):
        chunk_id = f"{doc_id}:{idx}"
        docs.append(
            IngestDoc(
                doc_id=doc_id,
                source_path=source_path,
                pdf_title=pdf_title,
                file_hash=file_hash,
                chunk_id=chunk_id,
                page_start=ch.page_start,
                page_end=ch.page_end,
                chunk_text=ch.text,
                created_at=created_at,
            )
        )
    return docs


def update_manifest_for_pdf(
    manifest: Dict[str, Dict[str, str]],
    pdf_path: Path,
    pdf_root: Path,
    file_hash: str,
    doc_id: str,
) -> None:
    manifest[manifest_key_for_pdf(pdf_path, pdf_root)] = {
        "file_hash": file_hash,
        "doc_id": doc_id,
        "updated_at": _utc_now_iso(),
    }


# Backward-compatible aliases for existing imports/callers.
iter_pdfs = iter_docs
ingest_pdf = ingest_document

