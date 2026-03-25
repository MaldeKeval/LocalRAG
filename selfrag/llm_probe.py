from __future__ import annotations

from dataclasses import dataclass

import requests

from .config import Settings

_REQUEST_TIMEOUT = 5.0


@dataclass(frozen=True)
class LlmProbeResult:
    """Result of checking whether the configured LLM is reachable and available."""

    ok: bool
    """False when the user should see a warning before generation."""
    message: str
    """Human-readable warning or detail; empty when ok is True."""
    status_line: str
    """Short line for CLI status table."""


def probe_llm(settings: Settings) -> LlmProbeResult:
    if settings.llm_provider == "ollama":
        return _probe_ollama(settings)
    return _probe_openai_compat(settings)


def _probe_ollama(settings: Settings) -> LlmProbeResult:
    base = settings.ollama_base_url.rstrip("/")
    model = (settings.ollama_model or "").strip()
    url = f"{base}/api/tags"

    try:
        r = requests.get(url, timeout=_REQUEST_TIMEOUT)
        r.raise_for_status()
    except requests.exceptions.RequestException as e:
        msg = (
            f"Cannot reach Ollama at {base}. Start Ollama or set ollama_base_url in "
            f"selfrag.config.json / SELF_RAG_OLLAMA_BASE_URL. ({e})"
        )
        return LlmProbeResult(
            ok=False,
            message=msg,
            status_line=f"Unreachable ({base})",
        )

    try:
        data = r.json()
    except Exception:
        return LlmProbeResult(
            ok=False,
            message=f"Unexpected response from Ollama at {url}. Check that Ollama is running.",
            status_line="Invalid response from /api/tags",
        )

    names: list[str] = []
    for m in data.get("models") or []:
        if isinstance(m, dict) and m.get("name"):
            names.append(str(m["name"]))

    if not model:
        return LlmProbeResult(
            ok=False,
            message="ollama_model is empty. Set ollama_model in selfrag.config.json.",
            status_line="No model configured",
        )

    if model in names:
        return LlmProbeResult(
            ok=True,
            message="",
            status_line=f"OK ({model})",
        )

    hint = (
        f"Model `{model}` is not available locally. Run `ollama pull {model}` or change "
        "ollama_model in selfrag.config.json to a model you have installed."
    )
    return LlmProbeResult(
        ok=False,
        message=hint,
        status_line=f"Missing: {model}",
    )


def _probe_openai_compat(settings: Settings) -> LlmProbeResult:
    base = settings.openai_compat_base_url.rstrip("/")
    model = (settings.openai_compat_model or "").strip()
    url = f"{base}/models"
    headers = {
        "Authorization": f"Bearer {settings.openai_compat_api_key}",
        "Content-Type": "application/json",
    }

    try:
        r = requests.get(url, headers=headers, timeout=_REQUEST_TIMEOUT)
    except requests.exceptions.RequestException as e:
        msg = (
            f"Cannot reach the OpenAI-compatible server at {base}. Start the server "
            f"(e.g. llama.cpp server) or set openai_compat_base_url / llama_server.base_url. ({e})"
        )
        return LlmProbeResult(
            ok=False,
            message=msg,
            status_line=f"Unreachable ({base})",
        )

    if r.status_code == 404:
        return LlmProbeResult(
            ok=True,
            message="",
            status_line="OK (model list unavailable)",
        )

    if not r.ok:
        return LlmProbeResult(
            ok=False,
            message=(
                f"OpenAI-compatible server returned HTTP {r.status_code} for {url}. "
                "Check the server and API key."
            ),
            status_line=f"HTTP {r.status_code}",
        )

    try:
        data = r.json()
    except Exception:
        return LlmProbeResult(
            ok=False,
            message=f"Could not parse JSON from {url}. Check the server.",
            status_line="Invalid /models response",
        )

    ids: list[str] = []
    for item in data.get("data") or []:
        if isinstance(item, dict) and item.get("id"):
            ids.append(str(item["id"]))

    if not model:
        return LlmProbeResult(
            ok=False,
            message="openai_compat_model is empty. Set it in selfrag.config.json (or llama_server.model).",
            status_line="No model configured",
        )

    if not ids:
        return LlmProbeResult(
            ok=True,
            message="",
            status_line="OK (empty model list; verify manually)",
        )

    if model in ids:
        return LlmProbeResult(
            ok=True,
            message="",
            status_line=f"OK ({model})",
        )

    hint = (
        f"Model `{model}` was not listed by the server at {url}. Change openai_compat_model "
        "(or llama_server.model) to a model your server exposes, or load that model in the server."
    )
    return LlmProbeResult(
        ok=False,
        message=hint,
        status_line=f"Missing: {model}",
    )
