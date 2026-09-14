"""The ``Embedder`` protocol — text -> vectors, per plan.md §6.

An embedder's job, precisely: ``list[str] -> np.ndarray[float32]``,
L2-normalised at the boundary, exposing ``dim`` and ``model_id``. Explicitly
not its job: storing anything, deciding chunk size, or retrying/rate-limiting
— local embedders need none of that (plan.md §4).

``FakeEmbedder`` (``tests/fakes.py``, built in Phase A) already matches this
shape, so it is this protocol's *second* implementation and B4 does not break
the "a Protocol is written at the second implementation" rule (plan.md §15
#1) — see plan.md §9 Phase B, session B4.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Protocol, runtime_checkable

import numpy as np


@runtime_checkable
class Embedder(Protocol):
    """Something that turns text into L2-normalised float32 vectors.

    Structural, not a base class — any object exposing ``dim``, ``model_id``
    and these two methods satisfies it.

    Attributes
    ----------
    dim
        Width of every vector this embedder produces.
    model_id
        Identifier of the embedding model, recorded alongside every vector so
        a store can refuse to mix two models in one index.

    """

    dim: int
    model_id: str

    def embed(self, texts: Sequence[str]) -> np.ndarray:
        """Return an ``(len(texts), dim)`` float32, L2-normalised array."""
        ...

    def embed_query(self, text: str) -> np.ndarray:
        """Return the ``(dim,)`` float32, L2-normalised query vector."""
        ...
