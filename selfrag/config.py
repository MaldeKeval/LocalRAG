from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Literal, Optional

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_prefix="SELF_RAG_",
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # Paths
    pdf_dir: Path = Field(default=Path("pdfs"))
    index_dir: Path = Field(default=Path("index"))

    # Chunking
    chunk_size: int = 1200
    chunk_overlap: int = 200
    min_chunk_size: int = 200

    # Retrieval
    top_k: int = 6
    hybrid_bm25: bool = False
    bm25_weight: float = 0.25

    # Embeddings
    embedding_model: str = "intfloat/e5-base-v2"
    embedding_device: Literal["auto", "cpu", "cuda"] = "auto"

    # Optional reranker
    reranker_model: Optional[str] = None  # e.g. "BAAI/bge-reranker-base"
    reranker_device: Literal["auto", "cpu", "cuda"] = "auto"

    # LLM provider
    llm_provider: Literal["ollama", "openai_compat"] = "ollama"
    ollama_base_url: str = "http://localhost:11434"
    ollama_model: str = "qwen2.5:7b"

    # llama.cpp server (or any OpenAI-compatible local server)
    openai_compat_base_url: str = "http://localhost:8080/v1"
    openai_compat_model: str = "qwen3.5-4b"
    openai_compat_api_key: str = "local"

    # Generation
    response_mode: Literal["strict", "blended"] = "strict"
    max_tokens: int = 512
    temperature: float = 0.2


def ensure_dirs(settings: Settings) -> None:
    settings.pdf_dir.mkdir(parents=True, exist_ok=True)
    settings.index_dir.mkdir(parents=True, exist_ok=True)


DEFAULT_JSON_CONFIG_PATH = Path("selfrag.config.json")


def load_settings(config_path: Optional[Path] = None) -> Settings:
    """
    Load settings with precedence:
      env vars (.env + process env) > JSON config file > code defaults
    """
    settings = Settings()

    path = config_path or DEFAULT_JSON_CONFIG_PATH
    if not path or not path.exists():
        return settings

    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return settings

    if not isinstance(data, dict):
        return settings

    # Aliases for JSON configs (not valid Literal values on Settings).
    _lp = data.get("llm_provider")
    if _lp in ("llama_server", "llama-server"):
        _ls = data.get("llama_server")
        if isinstance(_ls, dict) and not _ls.get("enabled", True):
            # Block disabled: use Ollama (OpenAI-compat URLs are not applied).
            data["llm_provider"] = "ollama"
        else:
            data["llm_provider"] = "openai_compat"

    # Optional nested block for llama.cpp server (OpenAI-compatible API).
    ls = data.get("llama_server")
    if isinstance(ls, dict) and ls.get("enabled", True):
        if "base_url" in ls:
            data["openai_compat_base_url"] = ls["base_url"]
        if "model" in ls:
            data["openai_compat_model"] = ls["model"]
        if "api_key" in ls:
            data["openai_compat_api_key"] = ls["api_key"]

    # Apply JSON as defaults without overriding environment-provided values.
    for key, value in data.items():
        if not isinstance(key, str):
            continue
        if not hasattr(settings, key):
            continue
        env_key = f"SELF_RAG_{key.upper()}"
        if os.getenv(env_key) is not None:
            continue
        try:
            setattr(settings, key, value)
        except Exception:
            continue

    return settings
