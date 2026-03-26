from __future__ import annotations

from typing import List

from .retrieval import RetrievedChunk


def _build_context(chunks: List[RetrievedChunk]) -> str:
    context_blocks: List[str] = []
    for i, c in enumerate(chunks, start=1):
        citation = f"{c.pdf_title} (p.{c.page_start}-{c.page_end})"
        context_blocks.append(f"[{i}] {citation}\n{c.chunk_text}".strip())
    return "\n\n---\n\n".join(context_blocks).strip()


def _strict_prompt(question: str, context: str) -> str:
    return (
        "You are a technical assistant answering questions about 3GPP specifications.\n"
        "Use ONLY the provided context. If the context does not contain the answer, say you don't know.\n"
        "Cite sources inline using [n] referencing the context blocks.\n\n"
        f"Question:\n{question.strip()}\n\n"
        f"Context:\n{context if context else '(no context)'}\n\n"
        "Answer (with citations [n]):"
    )


def _blended_prompt(question: str, context: str) -> str:
    return (
        "You are a technical assistant answering questions about 3GPP specifications.\n"
        "Use the provided context first, and separate grounded evidence from model prior knowledge.\n"
        "Follow this exact output format:\n"
        "Grounded answer:\n"
        "- Include only claims supported by the context.\n"
        "- Cite every factual claim inline using [n].\n"
        "- If the context is insufficient, explicitly say what is missing.\n\n"
        "Additional background (model knowledge, not from retrieved docs):\n"
        "- Optional section for helpful background.\n"
        "- Do not use [n] citations in this section.\n"
        "- Clearly mark uncertainty and avoid presenting guesses as facts.\n\n"
        "Do not mix uncited claims into the Grounded answer section.\n\n"
        f"Question:\n{question.strip()}\n\n"
        f"Context:\n{context if context else '(no context)'}\n\n"
        "Answer:"
    )


def build_prompt(question: str, chunks: List[RetrievedChunk], mode: str = "strict") -> str:
    context = _build_context(chunks)
    selected = (mode or "strict").strip().lower()
    if selected == "blended":
        return _blended_prompt(question, context)
    return _strict_prompt(question, context)


def build_grounded_prompt(question: str, chunks: List[RetrievedChunk]) -> str:
    # Backward-compatible wrapper for existing imports/call sites.
    return build_prompt(question, chunks, mode="strict")
