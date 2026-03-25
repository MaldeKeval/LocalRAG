from __future__ import annotations

from typing import List

from .retrieval import RetrievedChunk


def build_grounded_prompt(question: str, chunks: List[RetrievedChunk]) -> str:
    context_blocks: List[str] = []
    for i, c in enumerate(chunks, start=1):
        citation = f"{c.pdf_title} (p.{c.page_start}-{c.page_end})"
        context_blocks.append(f"[{i}] {citation}\n{c.chunk_text}".strip())

    context = "\n\n---\n\n".join(context_blocks).strip()

    # Keep the prompt simple and strict. Avoid leaking system text into model output.
    return (
        "You are a technical assistant answering questions about 3GPP specifications.\n"
        "Use ONLY the provided context. If the context does not contain the answer, say you don't know.\n"
        "Cite sources inline using [n] referencing the context blocks.\n\n"
        f"Question:\n{question.strip()}\n\n"
        f"Context:\n{context if context else '(no context)'}\n\n"
        "Answer (with citations [n]):"
    )
