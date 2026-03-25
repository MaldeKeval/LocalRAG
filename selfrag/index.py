from __future__ import annotations

from dataclasses import asdict
from typing import Dict, List

import lancedb
import numpy as np
import pyarrow as pa

from .config import Settings
from .embeddings import embed_texts
from .ingest import IngestDoc


def _schema(vector_dim: int) -> pa.Schema:
    return pa.schema(
        [
            pa.field("chunk_id", pa.string()),
            pa.field("doc_id", pa.string()),
            pa.field("source_path", pa.string()),
            pa.field("pdf_title", pa.string()),
            pa.field("file_hash", pa.string()),
            pa.field("page_start", pa.int32()),
            pa.field("page_end", pa.int32()),
            pa.field("chunk_text", pa.string()),
            pa.field("created_at", pa.string()),
            pa.field("embedding", pa.list_(pa.float32(), list_size=vector_dim)),
        ]
    )


class VectorIndex:
    def __init__(self, settings: Settings):
        self.settings = settings
        self.db_dir = settings.index_dir / "lancedb"
        self.db_dir.mkdir(parents=True, exist_ok=True)
        self.db = lancedb.connect(str(self.db_dir))

    def _get_or_create_table(self, vector_dim: int):
        if "chunks" in self.db.table_names():
            return self.db.open_table("chunks")
        return self.db.create_table("chunks", schema=_schema(vector_dim), mode="create")

    def upsert_docs(self, docs: List[IngestDoc]) -> None:
        if not docs:
            return

        texts = [d.chunk_text for d in docs]
        embeddings = embed_texts(self.settings.embedding_model, texts)
        vector_dim = int(embeddings.shape[1])
        tbl = self._get_or_create_table(vector_dim)

        # Remove old chunks for any doc_id in this batch (re-ingest replaces doc)
        doc_ids = sorted({d.doc_id for d in docs})
        for doc_id in doc_ids:
            try:
                tbl.delete(f"doc_id = '{doc_id}'")
            except Exception:
                # Table may be empty or filter may fail; ignore.
                pass

        rows = []
        for d, e in zip(docs, embeddings):
            r = asdict(d)
            r["embedding"] = e.tolist()
            rows.append(r)
        tbl.add(rows)

    def delete_doc_id(self, doc_id: str) -> None:
        if not doc_id:
            return
        if "chunks" not in self.db.table_names():
            return
        tbl = self.db.open_table("chunks")
        try:
            tbl.delete(f"doc_id = '{doc_id}'")
        except Exception:
            # If the filter fails or table is empty, treat as no-op.
            pass

    def count_chunks(self) -> int:
        if "chunks" not in self.db.table_names():
            return 0
        tbl = self.db.open_table("chunks")
        return int(tbl.count_rows())

    def unique_docs(self) -> int:
        if "chunks" not in self.db.table_names():
            return 0
        tbl = self.db.open_table("chunks")
        arrow = tbl.to_arrow()
        if arrow.num_rows == 0:
            return 0
        # Use Arrow compute to avoid needing pandas.
        unique = pa.compute.unique(arrow.column("doc_id"))
        return int(len(unique))

    def search(self, query_embedding: np.ndarray, top_k: int):
        if "chunks" not in self.db.table_names():
            return []
        tbl = self.db.open_table("chunks")
        return tbl.search(query_embedding.tolist()).limit(top_k).to_list()

    def all_chunks_for_bm25(self) -> List[Dict]:
        if "chunks" not in self.db.table_names():
            return []
        tbl = self.db.open_table("chunks")
        arrow = tbl.to_arrow()
        if arrow.num_rows == 0:
            return []
        return arrow.to_pylist()
