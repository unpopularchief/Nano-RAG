"""Reranking: re-score a candidate list, behind the ``Reranker`` protocol.

``IdentityReranker`` (the default, plan.md §6) has no dependencies and is
exported eagerly. ``JinaReranker`` (``httpx``) and
``LocalCrossEncoderReranker`` (``fastembed``/``onnxruntime``) are resolved
lazily on first attribute access, so ``import nanorag.rerank`` — and
therefore ``import nanorag``, which never reaches this far unless a caller
asks for one of them — stays free of both.
"""

from __future__ import annotations

from typing import Any

from nanorag.rerank.base import Reranker
from nanorag.rerank.identity import IdentityReranker

__all__ = [
    "Reranker",
    "IdentityReranker",
    "JinaReranker",
    "LocalCrossEncoderReranker",
]

_LAZY = {
    "JinaReranker": "nanorag.rerank.jina",
    "LocalCrossEncoderReranker": "nanorag.rerank.local_cross_encoder",
}


def __getattr__(name: str) -> Any:
    """Import a concrete reranker on first access (see the module docstring)."""
    if name in _LAZY:
        import importlib

        return getattr(importlib.import_module(_LAZY[name]), name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
