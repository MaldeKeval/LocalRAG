# SelfRag (Local PDF RAG)

![License](https://img.shields.io/badge/license-unknown-lightgrey)
![CI](https://img.shields.io/badge/CI-passing-brightgreen)
![Python](https://img.shields.io/badge/python-3.10%2B-blue)

SelfRag is a **fully local Retrieval-Augmented Generation (RAG)** tool: it searches your PDFs, retrieves the most relevant passages, and generates answers **grounded in those passages with citations** (PDF filename + page range).

## How to run

### Prerequisites

- **Python 3.10+**
- Optional (only needed for generation via `rag ask` / web UI):
  - **Ollama**, or
  - an **OpenAI-compatible local server** (e.g. llama.cpp `llama-server`)

### Setup

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -U pip
pip install -e .
```

### First run

Put PDFs in `pdfs/` (or pass a folder path), then ingest and ask:

```bash
rag ingest pdfs
rag status
rag ask "What is the purpose of the PDCP layer?"
```

### Web UI (optional)

```bash
rag-webui
```

## Tested models

These setups have been exercised with this project (your machine may differ):

| Provider | Models tested |
| --- | --- |
| **Ollama** | `qwen2.5:4b`, `qwen3.5:9b` |
| **llama-server** (OpenAI-compatible, e.g. llama.cpp) | `Qwen/Qwen2.5-7B-Instruct-GGUF:Q8_0` |

Use the same names in `selfrag.config.json` (`ollama_model` or `llama_server.model`) as your local install exposes.

## More docs

- `docs/architecture.md`
- `docs/configuration.md`
- `docs/llm-providers.md`
- `docs/project-status.md`
