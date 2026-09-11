"""Nano RAG — a small, readable retrieval-augmented generation engine.

The public surface grows one phase at a time (see ``plan.md`` §7). Phase A
exports the core value types and the error hierarchy; provider and embedder
classes are reached through ``nanorag.generation`` / ``nanorag.embeddings`` so
that ``import nanorag`` never pulls ``httpx``, ``fastembed`` or ``onnxruntime``.
"""

from nanorag.errors import (
    AuthError,
    ChunkingError,
    ConfigError,
    EmbeddingError,
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

__version__ = "0.0.1"

__all__ = [
    "__version__",
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
