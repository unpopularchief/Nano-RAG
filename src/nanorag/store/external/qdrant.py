"""``QdrantVectorStore`` — a ``VectorStore`` backed by a Qdrant collection.

``qdrant-client`` is imported lazily, inside ``__init__`` only, so importing
``nanorag.store.external`` never requires the ``[qdrant]`` extra — only
constructing a ``QdrantVectorStore`` does, mirroring
``embeddings.local.FastEmbedEmbedder`` (plan.md §3).

Point ids
---------
A ``nanorag`` ``chunk_id`` is a fixed-length sha256-hex string
(``hashing.chunk_id``), but Qdrant point ids must be an unsigned integer or a
UUID — an arbitrary hex string is rejected. Every chunk id is mapped to a
Qdrant point id via ``uuid.uuid5(_NAMESPACE, chunk_id)``: deterministic, so
both directions (``chunk_id -> point id`` for :meth:`upsert`/:meth:`delete`,
``point -> chunk_id`` for :meth:`search`/:meth:`get_vectors` results, read
back from the point's payload) need no separate mapping table and are stable
across restarts and processes.

Distance and filtering
-----------------------
The collection uses cosine distance. Vectors are already L2-normalised at
the embedder boundary (plan.md §6), so this is numerically identical to
``NumpyVectorStore``'s plain dot product. Metadata filters never reach this
class at all (see ``store.base`` module docstring) — :meth:`search` only
ever restricts to an id set, translated to a Qdrant ``MatchAny`` filter on
the ``chunk_id`` payload field.

Qdrant does not guarantee a tie-break order the way ``NumpyVectorStore``
does (ties broken by ascending chunk id); two candidates with an identical
score may come back in either order.
"""

from __future__ import annotations

import uuid
from collections.abc import Collection, Sequence
from typing import TYPE_CHECKING

import numpy as np

from nanorag.errors import ConfigError, StoreError

if TYPE_CHECKING:
    from qdrant_client import QdrantClient

#: Fixed constant so chunk-id -> point-id mapping is stable across restarts
#: and processes (arbitrary, generated once, never changed).
_NAMESPACE = uuid.UUID("6f5a1e00-6e5c-4c1b-8f3a-2f6c8b6a9d10")

#: Payload key the original chunk id is stored under, for the reverse
#: mapping back from a point to its chunk id.
_CHUNK_ID_KEY = "chunk_id"


def _point_id(chunk_id: str) -> str:
    """Return the deterministic Qdrant point id for *chunk_id*."""
    return str(uuid.uuid5(_NAMESPACE, chunk_id))


class QdrantVectorStore:
    """A ``VectorStore`` over a Qdrant collection.

    Parameters
    ----------
    dim
        Fixed vector width. The collection is created with this width if it
        does not already exist; an existing collection with a different
        width raises ``StoreError``.
    collection_name
        The Qdrant collection to use.
    url
        The Qdrant server URL.

    Raises
    ------
    ConfigError
        The ``[qdrant]`` extra is not installed.
    StoreError
        The server is unreachable, or an existing collection's vector width
        does not match *dim*.

    """

    def __init__(
        self,
        dim: int,
        *,
        collection_name: str = "nanorag",
        url: str = "http://localhost:6333",
    ) -> None:
        """Connect to *url* and ensure *collection_name* exists at width *dim*."""
        if dim < 1:
            raise StoreError(f"dim must be >= 1, got {dim}")

        try:
            from qdrant_client import QdrantClient
            from qdrant_client.models import Distance, VectorParams
        except ImportError as exc:
            raise ConfigError(
                "the Qdrant store needs the [qdrant] extra: run "
                '`uv add "nanorag[qdrant]"` (installs qdrant-client)'
            ) from exc

        client = QdrantClient(url=url)
        try:
            exists = client.collection_exists(collection_name)
            if not exists:
                client.create_collection(
                    collection_name,
                    vectors_config=VectorParams(size=dim, distance=Distance.COSINE),
                )
            else:
                info = client.get_collection(collection_name)
                existing_dim = info.config.params.vectors.size  # type: ignore[union-attr]
                if existing_dim != dim:
                    raise StoreError(
                        "existing Qdrant collection has a different vector width",
                        collection_name=collection_name,
                        existing_dim=existing_dim,
                        requested_dim=dim,
                    )
        except StoreError:
            raise
        except Exception as exc:
            raise StoreError(
                f"could not reach Qdrant at {url!r}: {exc}", url=url
            ) from exc

        self._dim = dim
        self._client: QdrantClient = client
        self._collection = collection_name

    @property
    def dim(self) -> int:
        """Fixed vector width of this store."""
        return self._dim

    def upsert(self, chunk_ids: Sequence[str], vectors: np.ndarray) -> None:
        """Insert new vectors and overwrite existing ones, by chunk id."""
        from qdrant_client.models import PointStruct

        vectors = np.asarray(vectors, dtype=np.float32)
        if vectors.ndim != 2 or vectors.shape[1] != self._dim:
            raise StoreError(
                "vectors must have shape (n, dim)",
                dim=self._dim,
                shape=tuple(vectors.shape),
            )
        if len(chunk_ids) != vectors.shape[0]:
            raise StoreError(
                "chunk_ids and vectors length mismatch",
                chunk_ids=len(chunk_ids),
                vectors=vectors.shape[0],
            )
        if len(chunk_ids) == 0:
            return

        points = [
            PointStruct(
                id=_point_id(cid), vector=vec.tolist(), payload={_CHUNK_ID_KEY: cid}
            )
            for cid, vec in zip(chunk_ids, vectors, strict=True)
        ]
        self._client.upsert(self._collection, points=points, wait=True)

    def delete(self, chunk_ids: Sequence[str]) -> None:
        """Remove *chunk_ids* from search. Unknown chunk ids are ignored."""
        from qdrant_client.models import PointIdsList

        if not chunk_ids:
            return
        self._client.delete(
            self._collection,
            points_selector=PointIdsList(points=[_point_id(cid) for cid in chunk_ids]),
            wait=True,
        )

    def get_vectors(self, chunk_ids: Sequence[str]) -> dict[str, np.ndarray]:
        """Return the stored (live) vectors for *chunk_ids*, keyed by id."""
        if not chunk_ids:
            return {}
        records = self._client.retrieve(
            self._collection,
            ids=[_point_id(cid) for cid in chunk_ids],
            with_vectors=True,
            with_payload=True,
        )
        found: dict[str, np.ndarray] = {}
        for record in records:
            assert record.payload is not None
            cid = record.payload[_CHUNK_ID_KEY]
            found[cid] = np.asarray(record.vector, dtype=np.float32)
        return found

    def search(
        self,
        query: np.ndarray,
        k: int,
        *,
        allowed_ids: Collection[str] | None = None,
    ) -> list[tuple[str, float]]:
        """Return up to *k* ``(chunk_id, score)`` pairs, highest score first."""
        from qdrant_client.models import FieldCondition, Filter, MatchAny

        query = np.asarray(query, dtype=np.float32)
        if query.shape != (self._dim,):
            raise StoreError(
                "query must have shape (dim,)",
                dim=self._dim,
                shape=tuple(query.shape),
            )
        if k < 1:
            raise StoreError(f"k must be >= 1, got {k}")
        if allowed_ids is not None and not allowed_ids:
            return []

        query_filter = None
        if allowed_ids is not None:
            query_filter = Filter(
                must=[
                    FieldCondition(
                        key=_CHUNK_ID_KEY, match=MatchAny(any=list(allowed_ids))
                    )
                ]
            )

        response = self._client.query_points(
            self._collection,
            query=query.tolist(),
            limit=k,
            query_filter=query_filter,
            with_payload=True,
        )
        results = []
        for point in response.points:
            assert point.payload is not None
            results.append((point.payload[_CHUNK_ID_KEY], float(point.score)))
        return results

    def compact(self) -> None:
        """No-op: Qdrant reclaims deleted-point storage on its own."""
