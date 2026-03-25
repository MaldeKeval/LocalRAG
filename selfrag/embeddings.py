from __future__ import annotations

import os
from functools import lru_cache
from typing import List

import numpy as np
from huggingface_hub.utils import logging as hub_logging
from sentence_transformers import SentenceTransformer
from transformers.utils import logging as hf_logging


@lru_cache(maxsize=2)
def _load_model(model_name: str) -> SentenceTransformer:
    # Reduce noisy HF/Transformers logging in CLI usage.
    os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")
    os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    os.environ.setdefault("TQDM_DISABLE", "1")
    hf_logging.set_verbosity_error()
    hub_logging.set_verbosity_error()

    # CPU default for stability on Windows; can be swapped later.
    return SentenceTransformer(model_name, device="cpu")


def embed_texts(model_name: str, texts: List[str]) -> np.ndarray:
    model = _load_model(model_name)
    emb = model.encode(
        texts,
        batch_size=32,
        show_progress_bar=False,
        convert_to_numpy=True,
        normalize_embeddings=True,
    )
    return np.asarray(emb, dtype=np.float32)


def embed_query(model_name: str, query: str) -> np.ndarray:
    return embed_texts(model_name, [query])[0]
