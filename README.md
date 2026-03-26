# SelfRag (Local 3GPP PDF RAG)

Fully local RAG. Search and ask questions across your specs with answers grounded in retrieved text and citations—no cloud embedding or retrieval required.

## Goals

- **Trust**: Answers cite PDF filename and page range.
- **Workflow**: CLI-first; optional Gradio web UI.
- **Offline**: local embeddings, vector store, and generation (Ollama or llama.cpp server).
- **MVP non-goals**: multi-user server deployment; heavy OCR pipelines (unless needed later).

## How it works

1. **Ingest**: PDFs → page text → normalized text → chunks with page ranges.
2. **Index**: Embed chunks; store vectors and metadata in LanceDB.
3. **Retrieve**: Embed the query; vector search (optional hybrid BM25); optional rerank.
4. **Generate**: Grounded prompt → local LLM → answer with citations.

**Incremental updates**: A per-PDF file hash avoids reprocessing unchanged PDFs; stable `doc_id` ties chunks back to source files.

## Stack

- **Python** 3.10+
- **PDFs**: PyMuPDF
- **Embeddings**: SentenceTransformers (CPU/GPU auto; E5 `query:`/`passage:` prefixing)
- **Vector DB**: LanceDB (embedded, local)
- **LLM**: Ollama or llama.cpp (OpenAI-compatible HTTP API)

## Status

The pipeline is implemented: incremental ingest, LanceDB index, retrieval (with optional hybrid BM25 / rerank), CLI (`ingest`, `status`, `ask`, `watch`, `open`), Gradio web UI, and local LLM providers. Optional lightweight retrieval eval: `python -m selfrag.eval --questions your_questions.jsonl` (each JSONL line: `id`, `question`, optional `must_cite` list of PDF basenames to check hit@k).

## Prerequisites

- **Python 3.10+**
- For **generation** (optional until you run `ask` / web UI with an LLM):
  - **Ollama** (recommended), or
  - **llama.cpp server** (OpenAI-compatible endpoint)

## Setup

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -U pip
pip install -e .
```

## Folder layout

- **`pdfs/`**: put PDFs here (or pass a folder to `rag ingest`)
- **`index/`**: persistent index (LanceDB + manifest)

## First run

```bash
rag ingest pdfs
rag status
rag ask "What is the purpose of the PDCP layer?"
rag ask --mode blended "Summarize PDCP and add missing background"
```

## Web UI (Gradio)

```bash
rag-webui
```

## Local LLM (generation)

- **Ollama**: ensure it is running, then pull a model (example):

```bash
ollama pull qwen2.5:7b
```

### Tested providers and models

These setups have been exercised with this project (your machine may differ):

| Provider | Models tested |
| --- | --- |
| **Ollama** | `qwen2.5:4b`, `qwen3.5:9b` |
| **llama-server** (OpenAI-compatible, e.g. llama.cpp) | `Qwen/Qwen2.5-7B-Instruct-GGUF:Q8_0` |

Use the same names in `selfrag.config.json` (`ollama_model` or `llama_server.model`) as your local install exposes.

### GGUF and `Q8_0`

- **GGUF** (GPT-Generated Unified Format) is a single-file container format for LLM weights, commonly used with **llama.cpp** and compatible servers. Quantized builds are often published as `.gguf` files or tagged on Hugging Face; the server loads one file and serves chat over an OpenAI-style API.
- **`Q8_0`** is a **quantization tag**: **8-bit** integer weights (`Q8`), with `_0` denoting a specific variant in the llama.cpp quantization naming scheme. It is typically higher fidelity than smaller quants (e.g. Q4) at the cost of larger files and more VRAM/RAM.

## Changing the LLM provider

Generation uses **`selfrag.config.json`** (repo root) or environment variables. **Precedence**: process env and `.env` override JSON; JSON overrides code defaults.

Supported **`llm_provider`** values:

| Value | Backend | When to use |
| --- | --- | --- |
| `ollama` | [Ollama](https://ollama.com/) HTTP API | Default; easiest on Windows. |
| `llama_server` or `llama-server` | Any **OpenAI-compatible** HTTP API (e.g. [llama.cpp server](https://github.com/ggerganov/llama.cpp)) | You run a local server that exposes `/v1/chat/completions`. |

Internally, `llama_server` / `llama-server` map to the same OpenAI-compatible client (`openai_compat`). If you use a `llama_server` JSON block with **`"enabled": false`**, the app treats generation as **`ollama`** (the nested URLs are ignored).

### Provider: `ollama`

1. Start Ollama (it listens on port **11434** by default).
2. Pull a model, e.g. `ollama pull qwen2.5:7b`.
3. In `selfrag.config.json` set:

```json
"llm_provider": "ollama",
"ollama_base_url": "http://localhost:11434",
"ollama_model": "qwen2.5:7b"
```

**Parameters**

| Key | Meaning |
| --- | --- |
| `ollama_base_url` | Ollama API base (no path suffix). |
| `ollama_model` | Model name as shown by `ollama list`. |

**Environment overrides** (optional): `SELF_RAG_LLM_PROVIDER=ollama`, `SELF_RAG_OLLAMA_BASE_URL`, `SELF_RAG_OLLAMA_MODEL`.

### Provider: `llama_server` (OpenAI-compatible server)

Use a **GGUF** build of your model (e.g. Qwen 2.5 from Hugging Face) with [llama.cpp `llama-server`](https://github.com/ggerganov/llama.cpp/blob/master/examples/server/README.md). The server must expose **`/v1/chat/completions`** (OpenAI-compatible).

1. Download a **`.gguf`** file (e.g. Qwen2.5-7B-Instruct quantized build).
2. Start the server on the host/port that matches `llama_server.base_url` (example: port **8033** to match the repo config):

```bash
llama-server -m path\to\your-model.q8_0.gguf --host 127.0.0.1 --port 8033
```

3. In `selfrag.config.json`, set `llm_provider` to `llama_server` and fill the `llama_server` block. The repository ships an example aligned with **Qwen 2.5 7B Instruct (GGUF)**—adjust `base_url`, `model`, and port if your setup differs:

```json
"llm_provider": "llama_server",
"llama_server": {
  "enabled": true,
  "base_url": "http://127.0.0.1:8033/v1",
  "model": "Qwen/Qwen2.5-7B-Instruct-GGUF:Q8_0",
  "api_key": "local"
},
"ollama_base_url": "http://localhost:11434",
"ollama_model": "qwen2.5:7b"
```

The `model` string must match what your server expects in chat requests (often the Hugging Face repo id and quant tag when using certain loaders). See the live values in **`selfrag.config.json`** at the repo root.

When `llama_server.enabled` is `true`, the nested block fills the OpenAI-compat settings: `base_url` → `openai_compat_base_url`, `model` → `openai_compat_model`, `api_key` → `openai_compat_api_key`.

**Parameters**

| Key | Meaning |
| --- | --- |
| `llama_server.enabled` | If `true`, apply the nested `base_url` / `model` / `api_key` to the OpenAI-compat client. |
| `llama_server.base_url` | Root URL including **`/v1`** (e.g. `http://127.0.0.1:8033/v1` as in `selfrag.config.json`, or `http://127.0.0.1:8080/v1`). |
| `llama_server.model` | Model id your server expects in the chat request (often matches the loaded weights name). |
| `llama_server.api_key` | Sent as `Authorization: Bearer …`; use a placeholder like `local` if the server does not check keys. |

You can also set **`llm_provider`** to **`openai_compat`** and configure `openai_compat_base_url`, `openai_compat_model`, and `openai_compat_api_key` directly at the top level of the JSON (same meaning as the `llama_server` block).

**Environment overrides** (optional): `SELF_RAG_LLM_PROVIDER=openai_compat`, `SELF_RAG_OPENAI_COMPAT_BASE_URL`, `SELF_RAG_OPENAI_COMPAT_MODEL`, `SELF_RAG_OPENAI_COMPAT_API_KEY`.

## Configuration (JSON)

You can configure embeddings, retrieval, chunking, and LLM settings via `selfrag.config.json` in the repo root.

- **Precedence**: environment variables (including `.env`) override `selfrag.config.json`, which overrides code defaults.
- **Other example fields**: `embedding_model`, `embedding_device`, `top_k`, `hybrid_bm25`, `chunk_size`, `reranker_model`, `reranker_device`, `max_tokens`, `temperature`, `response_mode`.

### Response modes

Use `response_mode` to control how answers combine retrieval context and model prior knowledge:

- `strict` (default): answer from retrieved context only. If context is missing, the model should say it does not know.
- `blended`: return two sections:
  - **Grounded answer**: claims from retrieved context with `[n]` citations.
  - **Additional background (model knowledge, not from retrieved docs)**: optional background clearly labeled as non-grounded.

You can set the default in `selfrag.config.json`:

```json
"response_mode": "strict"
```

You can also override per CLI request:

```bash
rag ask --mode strict "..."
rag ask --mode blended "..."
```

### Embeddings: E5 prefixing

If `embedding_model` is an **E5**-family model (e.g. `intfloat/e5-base-v2`), the app embeds using the recommended prefixes:

- Queries: `query: ...`
- PDF chunks: `passage: ...`

This typically improves retrieval quality.

### GPU usage (optional)

Embeddings and reranking default to **`auto`** device selection:

- If CUDA is available, use GPU.
- Otherwise, use CPU.

You can force behavior with:

- `embedding_device`: `"auto" | "cpu" | "cuda"`
- `reranker_device`: `"auto" | "cpu" | "cuda"`

If GPU initialization fails (common with mismatched PyTorch/CUDA installs), the app falls back to CPU.

## Notes

- Ingestion is incremental: new or changed PDFs are detected by file hash.
- Answers include citations (PDF filename + page range).

## Disclaimer

This project is provided **as is**, without warranty of any kind. The author(s) do not guarantee that the software will work correctly, remain available, or be free of defects. **Use it at your own risk.** The author(s) are not responsible for any damage, data loss, system failures, or other issues that may result from using this software.
