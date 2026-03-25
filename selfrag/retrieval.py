from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List

import numpy as np
from rank_bm25 import BM25Okapi

from .config import Settings
from .embeddings import embed_query
from .index import VectorIndex


@dataclass(frozen=True)
class RetrievedChunk:
    chunk_id: str
    doc_id: str
    source_path: str
    pdf_title: str
    page_start: int
    page_end: int
    chunk_text: str
    score: float


def _tokenize(text: str) -> List[str]:
    # Lightweight tokenizer; good enough for BM25 baseline.
    return [t for t in "".join(ch.lower() if ch.isalnum() else " " for ch in text).split() if t]


def _dense_only(settings: Settings, index: VectorIndex, query: str) -> List[RetrievedChunk]:
    q = embed_query(settings.embedding_model, query)
    rows = index.search(q, settings.top_k)
    out: List[RetrievedChunk] = []
    for r in rows:
        out.append(
            RetrievedChunk(
                chunk_id=r["chunk_id"],
                doc_id=r["doc_id"],
                source_path=r["source_path"],
                pdf_title=r.get("pdf_title") or "",
                page_start=int(r["page_start"]),
                page_end=int(r["page_end"]),
                chunk_text=r["chunk_text"],
                score=float(r.get("_distance") or r.get("_score") or 0.0),
            )
        )
    return out


def _hybrid_bm25(settings: Settings, index: VectorIndex, query: str) -> List[RetrievedChunk]:
     # For MVP: build BM25 over all chunks on demand.
     # For large corpora we’d persist a lexical index; this keeps design simple.
     all_rows = index.all_chunks_for_bm25()
     if not all_rows:
         return []
 
     corpus_tokens = [_tokenize(r["chunk_text"]) for r in all_rows]
     bm25 = BM25Okapi(corpus_tokens)
     q_tokens = _tokenize(query)
     scores = bm25.get_scores(q_tokens)
     bm25_rank = np.argsort(-scores)[: max(settings.top_k * 5, settings.top_k)]
 
     # Dense candidates
     dense = _dense_only(settings, index, query)
     dense_by_id = {d.chunk_id: d for d in dense}
 
     # Normalize BM25 into [0,1] to combine
     max_bm25 = float(np.max(scores)) if len(scores) else 1.0
     if max_bm25 <= 0:
         max_bm25 = 1.0
 
     combined: Dict[str, RetrievedChunk] = {}
     for i in bm25_rank:
         r = all_rows[int(i)]
         bm25_norm = float(scores[int(i)] / max_bm25)
         chunk_id = r["chunk_id"]
         base = dense_by_id.get(chunk_id)
         # Lance returns distance where smaller is better; convert to similarity-ish.
         dense_sim = 0.0
         if base is not None:
             d = float(base.score)
             dense_sim = 1.0 / (1.0 + max(d, 0.0))
 
         sim = (1.0 - settings.bm25_weight) * dense_sim + settings.bm25_weight * bm25_norm
         combined[chunk_id] = RetrievedChunk(
             chunk_id=chunk_id,
             doc_id=r["doc_id"],
             source_path=r["source_path"],
             pdf_title=r.get("pdf_title") or "",
             page_start=int(r["page_start"]),
             page_end=int(r["page_end"]),
             chunk_text=r["chunk_text"],
             score=float(sim),
         )
 
     # Ensure dense-only hits included
     for d in dense:
         if d.chunk_id not in combined:
             dd = float(d.score)
             dense_sim = 1.0 / (1.0 + max(dd, 0.0))
             combined[d.chunk_id] = RetrievedChunk(
                 chunk_id=d.chunk_id,
                 doc_id=d.doc_id,
                 source_path=d.source_path,
                 pdf_title=d.pdf_title,
                 page_start=d.page_start,
                 page_end=d.page_end,
                 chunk_text=d.chunk_text,
                 score=float(dense_sim),
             )
 
     ranked = sorted(combined.values(), key=lambda x: x.score, reverse=True)[: settings.top_k]
     return ranked


def retrieve(settings: Settings, index: VectorIndex, query: str) -> List[RetrievedChunk]:
    if settings.hybrid_bm25:
        return _hybrid_bm25(settings, index, query)
    return _dense_only(settings, index, query)
