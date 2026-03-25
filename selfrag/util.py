from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def load_json(path: Path, default: Any) -> Any:
    if not path.exists():
        return default
    return json.loads(path.read_text(encoding="utf-8"))


def save_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")


def normalize_path(p: Path) -> str:
    try:
        return str(p.resolve())
    except Exception:
        return str(p)


def manifest_key_for_pdf(pdf_path: Path, pdf_root: Path) -> str:
    """
    Stable key for manifests and stored citations: path relative to pdf_root,
    forward slashes only (no absolute paths or drive letters).
    """
    root = pdf_root.resolve()
    try:
        return pdf_path.resolve().relative_to(root).as_posix()
    except ValueError:
        return pdf_path.name
