from __future__ import annotations

from dataclasses import replace
from functools import lru_cache
from typing import List

from sentence_transformers import CrossEncoder

from .retrieval import RetrievedChunk


@lru_cache(maxsize=1)
def _load_reranker(model_name: str) -> CrossEncoder:
    # CPU for stability
    return CrossEncoder(model_name, device="cpu")


def maybe_rerank(model_name: str | None, query: str, chunks: List[RetrievedChunk]) -> List[RetrievedChunk]:
    if not model_name or not chunks:
        return chunks
    reranker = _load_reranker(model_name)
    pairs = [(query, c.chunk_text) for c in chunks]
    scores = reranker.predict(pairs)
    rescored = [replace(c, score=float(s)) for c, s in zip(chunks, scores)]
    return sorted(rescored, key=lambda x: x.score, reverse=True)
