from __future__ import annotations

import json
from dataclasses import dataclass

import requests

from .config import Settings


@dataclass(frozen=True)
class LlmResult:
    text: str


def generate(settings: Settings, prompt: str) -> LlmResult:
    if settings.llm_provider == "ollama":
        return _ollama_generate(settings, prompt)
    return _openai_compat_generate(settings, prompt)


def _ollama_text_from_generate_body(data: dict) -> str:
    """
    Qwen3+ "thinking" models on Ollama may leave `response` empty unless `think: false`
    is sent; some builds also expose chain-of-thought in `thinking`.
    """
    err = data.get("error")
    if err:
        raise RuntimeError(str(err))
    main = (data.get("response") or "").strip()
    if main:
        return main
    thinking = (data.get("thinking") or "").strip()
    if thinking:
        return thinking
    return ""


def _ollama_generate(settings: Settings, prompt: str) -> LlmResult:
    url = settings.ollama_base_url.rstrip("/") + "/api/generate"
    payload = {
        "model": settings.ollama_model,
        "prompt": prompt,
        "stream": False,
        # Thinking models (e.g. Qwen3): without this, the completion can be empty in `response`.
        "think": False,
        "options": {
            "temperature": settings.temperature,
            "num_predict": settings.max_tokens,
        },
    }
    try:
        r = requests.post(url, json=payload, timeout=600)
    except requests.exceptions.ConnectionError as e:
        raise RuntimeError(
            f"Cannot reach Ollama at {settings.ollama_base_url}. Start Ollama or fix ollama_base_url."
        ) from e
    r.raise_for_status()
    data = r.json()
    if not isinstance(data, dict):
        return LlmResult(text="")
    return LlmResult(text=_ollama_text_from_generate_body(data))


def _openai_compat_generate(settings: Settings, prompt: str) -> LlmResult:
    # llama.cpp server can expose OpenAI-style endpoint (/v1/chat/completions).
    url = settings.openai_compat_base_url.rstrip("/") + "/chat/completions"
    headers = {
        "Authorization": f"Bearer {settings.openai_compat_api_key}",
        "Content-Type": "application/json",
    }
    payload = {
        "model": settings.openai_compat_model,
        "messages": [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": prompt},
        ],
        "temperature": settings.temperature,
        "max_tokens": settings.max_tokens,
    }
    try:
        r = requests.post(url, headers=headers, json=payload, timeout=600)
    except requests.exceptions.ConnectionError as e:
        raise RuntimeError(
            f"Cannot reach OpenAI-compatible LLM at {settings.openai_compat_base_url}. "
            "Start the server (e.g. llama.cpp on that port) or set llm_provider to 'ollama' in selfrag.config.json."
        ) from e
    r.raise_for_status()
    data = r.json()
    text = ""
    try:
        text = data["choices"][0]["message"]["content"]
    except Exception:
        text = json.dumps(data)
    return LlmResult(text=(text or "").strip())
