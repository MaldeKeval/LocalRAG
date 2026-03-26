# Architecture and Design

## Goals

- **Trust**: Answers cite source filename and location range.
- **Workflow**: CLI-first; optional Gradio web UI.
- **Offline**: Local embeddings, vector store, and generation (Ollama or llama.cpp server).
- **MVP non-goals**: Multi-user server deployment; heavy OCR pipelines (unless needed later).

## How it works

1. **Ingest**: PDF/DOCX -> extracted text units -> normalized text -> chunks with ranges.
2. **Index**: Embed chunks; store vectors and metadata in LanceDB.
3. **Retrieve**: Embed the query; vector search (optional hybrid BM25); optional rerank.
4. **Generate**: Grounded prompt -> local LLM -> answer with citations.

Incremental updates use a per-document file hash to avoid reprocessing unchanged files. A stable `doc_id` ties chunks back to source files.

## Stack

- **Python** 3.10+
- **Document parsing**: PyMuPDF (PDF), python-docx (DOCX)
- **Embeddings**: SentenceTransformers (CPU/GPU auto; E5 `query:`/`passage:` prefixing)
- **Vector DB**: LanceDB (embedded, local)
- **LLM backends**: Ollama or llama.cpp server (OpenAI-compatible HTTP API)
