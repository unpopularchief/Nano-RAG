"""Nano RAG — a small, readable retrieval-augmented generation engine.

The public surface grows one phase at a time (see ``plan.md`` §7). The core
value types and the error hierarchy are exported eagerly. ``Rag`` (the
facade) is resolved lazily on first access — its stores need ``numpy`` —
and provider and embedder classes are reached through ``nanorag.generation``
/ ``nanorag.embeddings``, so ``import nanorag`` never pulls ``numpy``,
``httpx``, ``fastembed`` or ``onnxruntime``.
"""

from __future__ import annotations

from typing import Any

from nanorag.errors import (
    AuthError,
    ChunkingError,
    ConfigError,
    EmbeddingError,
    EvaluationError,
    GenerationError,
    IndexModelMismatch,
    LoaderError,
    NanoRagError,
    ProviderError,
    QuotaExhausted,
    RateLimitError,
    RetrievalError,
    StoreError,
    TransientError,
)
from nanorag.types import (
    Answer,
    Chunk,
    Citation,
    Document,
    IngestReport,
    JsonScalar,
    LoadIssue,
    ScoredChunk,
    Timings,
    Usage,
)

__version__ = "0.3.0"

__all__ = [
    "__version__",
    # facade (lazy)
    "Rag",
    # errors
    "NanoRagError",
    "ConfigError",
    "LoaderError",
    "ChunkingError",
    "EmbeddingError",
    "StoreError",
    "IndexModelMismatch",
    "RetrievalError",
    "GenerationError",
    "EvaluationError",
    "ProviderError",
    "AuthError",
    "RateLimitError",
    "TransientError",
    "QuotaExhausted",
    # types
    "Document",
    "Chunk",
    "ScoredChunk",
    "Citation",
    "Usage",
    "Timings",
    "Answer",
    "IngestReport",
    "LoadIssue",
    "JsonScalar",
]


def __getattr__(name: str) -> Any:
    """Resolve ``Rag`` on first access (see the module docstring)."""
    if name == "Rag":
        from nanorag.pipeline import Rag

        return Rag
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
