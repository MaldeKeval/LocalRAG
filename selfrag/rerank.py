from __future__ import annotations

from dataclasses import replace
from functools import lru_cache
from typing import List, Literal

from sentence_transformers import CrossEncoder

from .retrieval import RetrievedChunk


def _resolve_device(device: Literal["auto", "cpu", "cuda"]) -> Literal["cpu", "cuda"]:
    if device == "cpu":
        return "cpu"
    if device == "cuda":
        return "cuda"
    try:
        import torch

        return "cuda" if torch.cuda.is_available() else "cpu"
    except Exception:
        return "cpu"


@lru_cache(maxsize=4)
def _load_reranker(model_name: str, device: Literal["cpu", "cuda"]) -> CrossEncoder:
    return CrossEncoder(model_name, device=device)


def maybe_rerank(
    model_name: str | None,
    query: str,
    chunks: List[RetrievedChunk],
    *,
    device: Literal["auto", "cpu", "cuda"] = "auto",
) -> List[RetrievedChunk]:
    if not model_name or not chunks:
        return chunks
    resolved = _resolve_device(device)
    try:
        reranker = _load_reranker(model_name, resolved)
    except Exception:
        reranker = _load_reranker(model_name, "cpu")
    pairs = [(query, c.chunk_text) for c in chunks]
    scores = reranker.predict(pairs)
    rescored = [replace(c, score=float(s)) for c, s in zip(chunks, scores)]
    return sorted(rescored, key=lambda x: x.score, reverse=True)
