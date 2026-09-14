"""``FastEmbedEmbedder`` — the default local ONNX embedder (plan.md §4).

No torch, no network after the first model download, no rate limiting: the
whole point of the local-embeddings decision is that ingest needs none of
that (plan.md §4). ``fastembed``/``onnxruntime`` are imported lazily, inside
``__init__``, so ``import nanorag.embeddings`` never requires the ``[local]``
extra — only constructing a ``FastEmbedEmbedder`` does, and it raises
``ConfigError`` naming the exact install command if the extra is missing
(plan.md §3).
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING

import numpy as np

from nanorag.errors import ConfigError, EmbeddingError

if TYPE_CHECKING:
    from fastembed import TextEmbedding

#: Picked by reputation, not yet by the Phase D chunk-size/model sweep
#: (plan.md §4) — 768-dim, English-only (plan.md §18 F12).
DEFAULT_MODEL_ID = "BAAI/bge-base-en-v1.5"

#: Single-threaded ONNX Runtime execution avoids the small nondeterminism a
#: multi-threaded reduction order can introduce (plan.md §18 F5). Raise this
#: for throughput at the cost of that determinism guarantee.
DEFAULT_THREADS = 1


class FastEmbedEmbedder:
    """Local ONNX embeddings via ``fastembed``.

    Parameters
    ----------
    model_id
        A ``fastembed``-supported model name. Defaults to the plan's picked
        default, ``BAAI/bge-base-en-v1.5`` (768-dim, English).
    cache_dir
        Where ``fastembed`` caches downloaded model files. ``None`` uses
        ``fastembed``'s own resolution.
    threads
        ONNX Runtime intra/inter-op thread count. ``None`` uses
        ``fastembed``'s own default (higher throughput, small cross-run
        nondeterminism); the default here, ``1``, favours determinism.

    Raises
    ------
    ConfigError
        ``fastembed``/``onnxruntime`` are not installed, or *model_id* is not
        a model ``fastembed`` supports.

    """

    def __init__(
        self,
        model_id: str = DEFAULT_MODEL_ID,
        *,
        cache_dir: str | None = None,
        threads: int | None = DEFAULT_THREADS,
    ) -> None:
        """Resolve *model_id*'s dimension and construct the ONNX session."""
        try:
            from fastembed import TextEmbedding
        except ImportError as exc:
            raise ConfigError(
                "local embeddings need the [local] extra: run "
                '`uv add "nanorag[local]"` (installs fastembed + onnxruntime)'
            ) from exc

        try:
            self.dim = TextEmbedding.get_embedding_size(model_id)
        except ValueError as exc:
            raise ConfigError(
                f"{model_id!r} is not a fastembed-supported model",
                model_id=model_id,
            ) from exc

        self.model_id = model_id
        self._model: TextEmbedding = TextEmbedding(
            model_name=model_id, cache_dir=cache_dir, threads=threads
        )

    def embed(self, texts: Sequence[str]) -> np.ndarray:
        """Return an ``(len(texts), dim)`` float32, L2-normalised array."""
        if len(texts) == 0:
            return np.empty((0, self.dim), dtype=np.float32)
        try:
            vectors = list(self._model.embed(list(texts)))
        except Exception as exc:
            # fastembed/onnxruntime raise a variety of undocumented exception
            # types across versions and backends; normalise all of them to
            # our own error hierarchy at this boundary.
            raise EmbeddingError(f"local embedding failed: {exc}") from exc
        return _l2_normalize(np.asarray(vectors, dtype=np.float32))

    def embed_query(self, text: str) -> np.ndarray:
        """Return the ``(dim,)`` float32, L2-normalised query vector."""
        try:
            (vector,) = list(self._model.query_embed([text]))
        except Exception as exc:
            raise EmbeddingError(f"local embedding failed: {exc}") from exc
        return _l2_normalize(np.asarray(vector, dtype=np.float32))


def _l2_normalize(vectors: np.ndarray) -> np.ndarray:
    """L2-normalise *vectors* along the last axis (plan.md §6 boundary rule).

    Works for both a single ``(dim,)`` vector and a ``(n, dim)`` batch. A
    zero vector (e.g. embedding an empty string) is left as-is rather than
    dividing by zero.
    """
    norms = np.linalg.norm(vectors, axis=-1, keepdims=True)
    norms = np.where(norms == 0, 1.0, norms)
    return (vectors / norms).astype(np.float32)
