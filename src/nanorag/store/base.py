"""The ``VectorStore`` protocol — upsert/delete/search/get_vectors/compact.

Written now, at the third implementation (plan.md §15 #1): ``NumpyVectorStore``
(Phase B) and the Phase G session G2 external adapters (``store.external.
qdrant.QdrantVectorStore``, ``store.external.pgvector.PgVectorStore``) all
satisfy this one shape, so ``Rag``, ``DenseRetriever`` and ``MmrRetriever``
can hold any of them behind the same attribute — "swapping the store changes
one constructor line" (plan.md §9 Phase G session G2).

This is exactly ``NumpyVectorStore``'s existing public surface (plan.md §6:
"upsert, search, delete, compact"); nothing was added or dropped to write
this protocol. Metadata filters never reach a ``VectorStore`` at all — a
``Filter`` is always resolved to an ``allowed_ids`` set via
``SqliteDocumentStore.filter_chunk_ids`` before ``search`` is ever called
(see ``retrieval/dense.py``) — so every implementation, external or not,
only ever needs an id-set restriction, never the filter grammar itself.
"""

from __future__ import annotations

from collections.abc import Collection, Sequence
from typing import Protocol, runtime_checkable

import numpy as np


@runtime_checkable
class VectorStore(Protocol):
    """Structural protocol: any object with this shape satisfies it."""

    @property
    def dim(self) -> int:
        """Fixed vector width this store is configured for."""
        ...

    def upsert(self, chunk_ids: Sequence[str], vectors: np.ndarray) -> None:
        """Insert new vectors and overwrite existing ones, by chunk id.

        Overwriting a chunk id that was previously deleted revives it.
        """
        ...

    def delete(self, chunk_ids: Sequence[str]) -> None:
        """Remove *chunk_ids* from search. Unknown chunk ids are ignored."""
        ...

    def search(
        self,
        query: np.ndarray,
        k: int,
        *,
        allowed_ids: Collection[str] | None = None,
    ) -> list[tuple[str, float]]:
        """Return up to *k* ``(chunk_id, score)`` pairs, highest score first.

        *query* is a ``(dim,)`` vector, already L2-normalised.
        *allowed_ids*, when given, restricts the search to this set of
        chunk ids (a metadata pre-filter's result), expressed in id space.
        """
        ...

    def get_vectors(self, chunk_ids: Sequence[str]) -> dict[str, np.ndarray]:
        """Return the stored (live) vectors for *chunk_ids*, keyed by id.

        Ids that are not stored, or are deleted, are simply absent from the
        result.
        """
        ...

    def compact(self) -> None:
        """Reclaim space freed by deletions, if this backend needs it.

        A no-op for backends that reclaim space on their own (both external
        adapters); required for ``NumpyVectorStore``, whose deletes only
        tombstone rows until this is called.
        """
        ...
