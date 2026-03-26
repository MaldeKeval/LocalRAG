from __future__ import annotations

from dataclasses import asdict
from pathlib import Path
import re
from typing import Any, List, Optional, Tuple

import gradio as gr

from .config import Settings, ensure_dirs, load_settings
from .index import VectorIndex
from .llm import generate
from .llm_probe import probe_llm
from .prompting import build_grounded_prompt
from .rerank import maybe_rerank
from .retrieval import RetrievedChunk, retrieve


def _citations_rows(chunks: List[RetrievedChunk]) -> List[List[Any]]:
    rows: List[List[Any]] = []
    for i, c in enumerate(chunks, start=1):
        rows.append(
            [
                i,
                Path(c.source_path).name,
                f"{c.page_start}-{c.page_end}",
                c.pdf_title or "",
                float(c.score),
            ]
        )
    return rows


_INLINE_LATEX_RE = re.compile(r"\\\((.+?)\\\)", flags=re.DOTALL)
_DISPLAY_LATEX_RE = re.compile(r"\\\[(.+?)\\\]", flags=re.DOTALL)


def _normalize_latex_delimiters(text: str) -> str:
    """
    Gradio markdown treats backslashes as escapes, which can break LaTeX delimiters
    like \\( ... \\) and \\[ ... \\]. Normalize them to $...$ / $$...$$ so KaTeX
    can render reliably.
    """
    if not text:
        return text
    text = _DISPLAY_LATEX_RE.sub(r"$$\1$$", text)
    text = _INLINE_LATEX_RE.sub(r"$\1$", text)
    return text


def answer_question(
    question: str,
    top_k: int,
    hybrid: bool,
    no_llm: bool,
) -> Tuple[str, List[List[Any]], str]:
    settings = load_settings()
    ensure_dirs(settings)
    settings.top_k = int(top_k)
    settings.hybrid_bm25 = bool(hybrid)

    q = (question or "").strip()
    if not q:
        return "Please enter a question.", [], ""

    index = VectorIndex(settings)
    chunks = retrieve(settings, index, q)
    chunks = maybe_rerank(
        settings.reranker_model,
        q,
        chunks,
        device=getattr(settings, "reranker_device", "auto"),
    )

    if no_llm:
        answer_text = "LLM skipped. Showing retrieved evidence only."
    else:
        probe = probe_llm(settings)
        prefix = ""
        if not probe.ok:
            prefix = f"**Warning:** {probe.message}\n\n---\n\n"
        prompt = build_grounded_prompt(q, chunks)
        result = generate(settings, prompt)
        body = _normalize_latex_delimiters((result.text or "").strip()) or "(empty response)"
        answer_text = prefix + body

    debug_json = ""
    try:
        debug_json = "\n".join(
            [
                str(
                    {
                        "question": q,
                        "settings": {
                            "top_k": settings.top_k,
                            "hybrid_bm25": settings.hybrid_bm25,
                            "bm25_weight": settings.bm25_weight,
                            "embedding_model": settings.embedding_model,
                            "reranker_model": settings.reranker_model,
                            "llm_provider": settings.llm_provider,
                            "ollama_model": settings.ollama_model,
                            "openai_compat_model": settings.openai_compat_model,
                        },
                        "chunks": [asdict(c) for c in chunks],
                    }
                )
            ]
        )
    except Exception:
        debug_json = ""

    return answer_text, _citations_rows(chunks), debug_json


def build_ui() -> gr.Blocks:
    with gr.Blocks(title="SelfRag") as demo:
        gr.Markdown(
            "## SelfRag (Local PDF RAG)\n"
            "Ask a question grounded in your ingested PDFs. Answers include citations."
        )

        with gr.Row():
            question = gr.Textbox(
                label="Question",
                lines=4,
                placeholder="Ask a question about your 3GPP PDFs…",
            )

        with gr.Row():
            ask_btn = gr.Button("Ask", variant="primary")
            clear_btn = gr.Button("Clear")

        with gr.Accordion("Advanced", open=False):
            top_k = gr.Slider(
                minimum=1,
                maximum=25,
                value=load_settings().top_k,
                step=1,
                label="top_k (retrieval)",
            )
            hybrid = gr.Checkbox(value=load_settings().hybrid_bm25, label="Hybrid BM25 retrieval")
            no_llm = gr.Checkbox(value=False, label="Skip LLM (evidence only)")

        answer = gr.Markdown(
            label="Answer",
            latex_delimiters=[
                {"left": "$$", "right": "$$", "display": True},
                {"left": "$", "right": "$", "display": False},
            ],
        )
        citations = gr.Dataframe(
            headers=["#", "PDF", "Pages", "Title", "Score"],
            datatype=["number", "str", "str", "str", "number"],
            label="Citations",
            interactive=False,
            wrap=True,
        )
        debug = gr.Textbox(label="Debug (raw)", lines=8, visible=False)

        ask_btn.click(
            fn=answer_question,
            inputs=[question, top_k, hybrid, no_llm],
            outputs=[answer, citations, debug],
        )
        question.submit(
            fn=answer_question,
            inputs=[question, top_k, hybrid, no_llm],
            outputs=[answer, citations, debug],
        )

        def _clear() -> Tuple[str, List[List[Any]], str]:
            return "", [], ""

        clear_btn.click(fn=_clear, inputs=[], outputs=[question, citations, answer])

    return demo


def main(host: Optional[str] = None, port: Optional[int] = None) -> None:
    demo = build_ui()
    demo.launch(
        server_name=host or "127.0.0.1",
        server_port=port,
        inbrowser=True,
    )

