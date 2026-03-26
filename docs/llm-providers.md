# LLM Providers

Generation uses `selfrag.config.json` (repository root) or environment variables.

## Provider selection

Supported `llm_provider` values:

| Value | Backend | When to use |
| --- | --- | --- |
| `ollama` | [Ollama](https://ollama.com/) HTTP API | Default; easiest on Windows. |
| `llama_server` or `llama-server` | OpenAI-compatible HTTP API (for example llama.cpp server) | You run a local server exposing `/v1/chat/completions`. |

Internally, `llama_server` and `llama-server` map to the same OpenAI-compatible client (`openai_compat`). If the `llama_server` block has `"enabled": false`, generation falls back to Ollama.

## Provider: Ollama

1. Start Ollama (default port `11434`).
2. Pull a model, for example:

```bash
ollama pull qwen2.5:7b
```

3. Set config values:

```json
"llm_provider": "ollama",
"ollama_base_url": "http://localhost:11434",
"ollama_model": "qwen2.5:7b"
```

Environment overrides (optional):

- `SELF_RAG_LLM_PROVIDER=ollama`
- `SELF_RAG_OLLAMA_BASE_URL`
- `SELF_RAG_OLLAMA_MODEL`

## Provider: llama_server (OpenAI-compatible)

Use a GGUF build of your model with llama.cpp `llama-server`. The server must expose `/v1/chat/completions`.

1. Download a `.gguf` model file.
2. Start server (example):

```bash
llama-server -m path\to\your-model.q8_0.gguf --host 127.0.0.1 --port 8033
```

3. Configure provider:

```json
"llm_provider": "llama_server",
"llama_server": {
  "enabled": true,
  "base_url": "http://127.0.0.1:8033/v1",
  "model": "Qwen/Qwen2.5-7B-Instruct-GGUF:Q8_0",
  "api_key": "local"
}
```

When `llama_server.enabled` is `true`, nested values map to OpenAI-compatible fields:

- `base_url` -> `openai_compat_base_url`
- `model` -> `openai_compat_model`
- `api_key` -> `openai_compat_api_key`

You can also set `llm_provider` to `openai_compat` and use top-level OpenAI-compatible fields directly.

Environment overrides (optional):

- `SELF_RAG_LLM_PROVIDER=openai_compat`
- `SELF_RAG_OPENAI_COMPAT_BASE_URL`
- `SELF_RAG_OPENAI_COMPAT_MODEL`
- `SELF_RAG_OPENAI_COMPAT_API_KEY`

## GGUF and Q8_0

- **GGUF** is a common single-file container format for model weights used by llama.cpp and compatible servers.
- **Q8_0** is an 8-bit quantization variant. It usually offers better quality than smaller quantizations, with higher RAM/VRAM and disk requirements.
