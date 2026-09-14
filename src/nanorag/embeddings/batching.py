"""Bounding how many texts reach an embedder in one call.

This is the one place any concurrency for embedding would live (plan.md §15
#21: "Async creep | Sync core is a stated invariant; concurrency lives in one
batching module."). It stays synchronous here: local embedding is CPU/ONNX
bound with no rate limit to respect, so there is nothing yet for a bounded
``ThreadPoolExecutor`` to buy. The seam exists for Phase G's hosted embedding
backends, which will need one.
"""

from __future__ import annotations

from collections.abc import Iterator, Sequence

import numpy as np

from nanorag.embeddings.base import Embedder

#: Matches ``config.Settings.embed_batch_size``'s default.
DEFAULT_BATCH_SIZE = 64


def batched(items: Sequence[str], batch_size: int) -> Iterator[Sequence[str]]:
    """Yield *items* in consecutive slices of at most *batch_size*.

    Parameters
    ----------
    items
        The items to split.
    batch_size
        Maximum length of each yielded slice.

    Raises
    ------
    ValueError
        *batch_size* is less than 1.

    """
    if batch_size < 1:
        raise ValueError(f"batch_size must be >= 1, got {batch_size!r}")
    for start in range(0, len(items), batch_size):
        yield items[start : start + batch_size]


class BatchingEmbedder:
    """Wraps an :class:`Embedder`, calling ``embed`` in bounded batches.

    Bounds peak memory/CPU for a large ``embed(texts)`` call, independent of
    whatever batching (if any) the wrapped embedder already does internally.

    Parameters
    ----------
    embedder
        The embedder to wrap.
    batch_size
        Maximum number of texts passed to the wrapped embedder's ``embed``
        in one call.

    Raises
    ------
    ValueError
        *batch_size* is less than 1.

    """

    def __init__(
        self, embedder: Embedder, *, batch_size: int = DEFAULT_BATCH_SIZE
    ) -> None:
        """Wrap *embedder*, bounding its ``embed`` calls to *batch_size*."""
        if batch_size < 1:
            raise ValueError(f"batch_size must be >= 1, got {batch_size!r}")
        self._embedder = embedder
        self._batch_size = batch_size
        self.dim = embedder.dim
        self.model_id = embedder.model_id

    def embed(self, texts: Sequence[str]) -> np.ndarray:
        """Return an ``(len(texts), dim)`` array, computed in bounded batches."""
        texts = list(texts)
        if not texts:
            return np.empty((0, self.dim), dtype=np.float32)
        parts = [
            self._embedder.embed(batch) for batch in batched(texts, self._batch_size)
        ]
        return np.concatenate(parts, axis=0)

    def embed_query(self, text: str) -> np.ndarray:
        """Delegate to the wrapped embedder — one query needs no batching."""
        return self._embedder.embed_query(text)
