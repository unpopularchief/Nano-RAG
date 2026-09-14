"""Turns chunk text into vectors: local ONNX by default, cached, batched.

Reached via ``nanorag.embeddings``, not the top level (plan.md §7 "Import
cost"), matching ``nanorag.loaders``/``nanorag.chunking``/``nanorag.store`` —
``import nanorag`` must stay free of ``fastembed``/``onnxruntime``/``numpy``.
"""

from nanorag.embeddings.base import Embedder
from nanorag.embeddings.batching import DEFAULT_BATCH_SIZE, BatchingEmbedder, batched
from nanorag.embeddings.cache import CachingEmbedder, EmbeddingCache, fingerprint
from nanorag.embeddings.local import DEFAULT_MODEL_ID, FastEmbedEmbedder

__all__ = [
    "Embedder",
    "FastEmbedEmbedder",
    "DEFAULT_MODEL_ID",
    "EmbeddingCache",
    "CachingEmbedder",
    "fingerprint",
    "BatchingEmbedder",
    "batched",
    "DEFAULT_BATCH_SIZE",
]
