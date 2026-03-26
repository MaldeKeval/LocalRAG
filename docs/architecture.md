# Architecture and Design

## Goals

- **Trust**: Answers cite PDF filename and page range.
- **Workflow**: CLI-first; optional Gradio web UI.
- **Offline**: Local embeddings, vector store, and generation (Ollama or llama.cpp server).
- **MVP non-goals**: Multi-user server deployment; heavy OCR pipelines (unless needed later).

## How it works

1. **Ingest**: PDFs -> page text -> normalized text -> chunks with page ranges.
2. **Index**: Embed chunks; store vectors and metadata in LanceDB.
3. **Retrieve**: Embed the query; vector search (optional hybrid BM25); optional rerank.
4. **Generate**: Grounded prompt -> local LLM -> answer with citations.

Incremental updates use a per-PDF file hash to avoid reprocessing unchanged PDFs. A stable `doc_id` ties chunks back to source files.

## Stack

- **Python** 3.10+
- **PDF parsing**: PyMuPDF
- **Embeddings**: SentenceTransformers (CPU/GPU auto; E5 `query:`/`passage:` prefixing)
- **Vector DB**: LanceDB (embedded, local)
- **LLM backends**: Ollama or llama.cpp server (OpenAI-compatible HTTP API)
