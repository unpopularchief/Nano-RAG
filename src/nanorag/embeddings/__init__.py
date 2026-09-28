"""Turns chunk text into vectors: local ONNX by default, cached, batched.

Reached via ``nanorag.embeddings``, not the top level (plan.md §7 "Import
cost"), matching ``nanorag.loaders``/``nanorag.chunking``/``nanorag.store`` —
``import nanorag`` must stay free of ``fastembed``/``onnxruntime``/``numpy``.
Hosted API clients are resolved lazily so importing the pipeline stays
``httpx``-free too.
"""

from typing import TYPE_CHECKING, Any

from nanorag.embeddings.base import Embedder
from nanorag.embeddings.batching import DEFAULT_BATCH_SIZE, BatchingEmbedder, batched
from nanorag.embeddings.cache import CachingEmbedder, EmbeddingCache, fingerprint
from nanorag.embeddings.local import DEFAULT_MODEL_ID, FastEmbedEmbedder

if TYPE_CHECKING:
    from nanorag.embeddings.api import (
        GeminiEmbedder,
        JinaEmbedder,
        OpenAICompatEmbedder,
    )

__all__ = [
    "Embedder",
    "FastEmbedEmbedder",
    "GeminiEmbedder",
    "JinaEmbedder",
    "OpenAICompatEmbedder",
    "DEFAULT_MODEL_ID",
    "EmbeddingCache",
    "CachingEmbedder",
    "fingerprint",
    "BatchingEmbedder",
    "batched",
    "DEFAULT_BATCH_SIZE",
]

_LAZY = {
    "GeminiEmbedder": "nanorag.embeddings.api",
    "JinaEmbedder": "nanorag.embeddings.api",
    "OpenAICompatEmbedder": "nanorag.embeddings.api",
}


def __getattr__(name: str) -> Any:
    """Import hosted clients only when a caller requests one."""
    if name in _LAZY:
        import importlib

        return getattr(importlib.import_module(_LAZY[name]), name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
