# Configuration

You can configure embeddings, retrieval, chunking, and LLM settings via `selfrag.config.json` in the repository root.

## Precedence

Environment variables (including `.env`) override `selfrag.config.json`, which overrides code defaults.

## Common fields

Examples include:

- `embedding_model`
- `embedding_device`
- `top_k`
- `hybrid_bm25`
- `chunk_size`
- `reranker_model`
- `reranker_device`
- `max_tokens`
- `temperature`
- `response_mode`

## Response modes

Use `response_mode` to control how answers combine retrieval context and model prior knowledge:

- `strict` (default): answer from retrieved context only. If context is missing, the model should say it does not know.
- `blended`: return two sections:
  - Grounded answer: claims from retrieved context with `[n]` citations.
  - Additional background: model knowledge not from retrieved docs, clearly labeled as non-grounded.

Set default in JSON:

```json
"response_mode": "strict"
```

Override per CLI request:

```bash
rag ask --mode strict "..."
rag ask --mode blended "..."
```

## Embeddings: E5 prefixing

If `embedding_model` is an E5-family model (for example `intfloat/e5-base-v2`), embedding uses recommended prefixes:

- Queries: `query: ...`
- Document chunks: `passage: ...`

This typically improves retrieval quality.

## GPU usage (optional)

Embeddings and reranking default to `auto` device selection:

- If CUDA is available, use GPU.
- Otherwise, use CPU.

You can force behavior with:

- `embedding_device`: `"auto" | "cpu" | "cuda"`
- `reranker_device`: `"auto" | "cpu" | "cuda"`

If GPU initialization fails (common with mismatched PyTorch/CUDA installs), the app falls back to CPU.
