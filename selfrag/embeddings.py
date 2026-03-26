from __future__ import annotations

import os
from functools import lru_cache
from typing import List, Literal

import numpy as np
from huggingface_hub.utils import logging as hub_logging
from sentence_transformers import SentenceTransformer
from transformers.utils import logging as hf_logging


def _is_e5_model(model_name: str) -> bool:
    # Heuristic: works for common HF IDs like "intfloat/e5-base-v2".
    return "e5" in (model_name or "").lower()


def _with_prefix(kind: Literal["query", "passage"], text: str) -> str:
    t = (text or "").strip()
    # E5-style prompting for embeddings.
    if kind == "query":
        return f"query: {t}"
    return f"passage: {t}"


def _resolve_device(device: Literal["auto", "cpu", "cuda"]) -> Literal["cpu", "cuda"]:
    if device == "cpu":
        return "cpu"
    if device == "cuda":
        return "cuda"
    try:
        import torch

        return "cuda" if torch.cuda.is_available() else "cpu"
    except Exception:
        return "cpu"


@lru_cache(maxsize=4)
def _load_model(model_name: str, device: Literal["cpu", "cuda"]) -> SentenceTransformer:
    # Reduce noisy HF/Transformers logging in CLI usage.
    os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")
    os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    os.environ.setdefault("TQDM_DISABLE", "1")
    hf_logging.set_verbosity_error()
    hub_logging.set_verbosity_error()

    return SentenceTransformer(model_name, device=device)


def embed_texts(
    model_name: str,
    texts: List[str],
    *,
    kind: Literal["query", "passage"] = "passage",
    device: Literal["auto", "cpu", "cuda"] = "auto",
) -> np.ndarray:
    resolved = _resolve_device(device)
    try:
        model = _load_model(model_name, resolved)
    except Exception:
        # If CUDA load fails (common on Windows with mismatched torch), retry on CPU.
        model = _load_model(model_name, "cpu")

    to_embed = texts
    if _is_e5_model(model_name):
        to_embed = [_with_prefix(kind, t) for t in texts]

    emb = model.encode(
        to_embed,
        batch_size=32,
        show_progress_bar=False,
        convert_to_numpy=True,
        normalize_embeddings=True,
    )
    return np.asarray(emb, dtype=np.float32)


def embed_query(
    model_name: str,
    query: str,
    *,
    device: Literal["auto", "cpu", "cuda"] = "auto",
) -> np.ndarray:
    return embed_texts(model_name, [query], kind="query", device=device)[0]
