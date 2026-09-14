"""``DenseRetriever`` — exact top-*k* cosine retrieval with metadata pre-filtering.

The retriever's job, per plan.md §6: query → ranked ``ScoredChunk``s, applying
metadata **pre**-filters. Not its job: prompting or generation. The whole read
path for one query is four calls you can follow by reading — embed the query,
resolve the filter to chunk ids in SQLite, search the vector store over those
ids, hydrate the hits back into ``Chunk``s.

Why *pre*-filtering: taking the top-*k* first and discarding non-matching
hits afterwards silently returns fewer than *k* results — or none — whenever
the best matches for the query happen to sit outside the filter, even though
the corpus holds *k* perfectly good matching chunks (plan.md §15 #7; proven by
``tests/test_dense_retriever.py``).
"""

from __future__ import annotations

from nanorag.embeddings.base import Embedder
from nanorag.errors import RetrievalError
from nanorag.store.filters import Filter
from nanorag.store.numpy_store import NumpyVectorStore
from nanorag.store.sqlite_docs import SqliteDocumentStore
from nanorag.types import ScoredChunk

#: The ``ScoredChunk.source`` every hit from this retriever carries.
SOURCE = "dense"


class DenseRetriever:
    """Exact cosine top-*k* over a ``NumpyVectorStore``, pre-filtered via SQLite.

    Parameters
    ----------
    embedder
        Produces the query vector. Must be the same model the index was
        built with — its ``dim`` is checked against the vector store here;
        ``SqliteDocumentStore`` refuses a second ``model_id`` at ingest time.
    vectors
        The in-memory index to search.
    docs
        The document store the index was built from; resolves filters and
        turns hits back into ``Chunk`` objects.
    k
        Default number of results when ``retrieve`` is not given one.

    Raises
    ------
    RetrievalError
        ``embedder.dim`` does not match ``vectors.dim``, or *k* < 1.

    """

    def __init__(
        self,
        embedder: Embedder,
        vectors: NumpyVectorStore,
        docs: SqliteDocumentStore,
        *,
        k: int = 10,
    ) -> None:
        """Wire the three stores together; validate the dimension and *k*."""
        if embedder.dim != vectors.dim:
            raise RetrievalError(
                "embedder and vector store dimensions differ",
                embedder_dim=embedder.dim,
                store_dim=vectors.dim,
                model_id=embedder.model_id,
            )
        if k < 1:
            raise RetrievalError(f"k must be >= 1, got {k}")
        self._embedder = embedder
        self._vectors = vectors
        self._docs = docs
        self._k = k

    @property
    def k(self) -> int:
        """Default number of results."""
        return self._k

    def retrieve(
        self,
        query: str,
        k: int | None = None,
        filter: Filter | None = None,
    ) -> list[ScoredChunk]:
        """Return the top-*k* chunks for *query*, highest cosine first.

        Parameters
        ----------
        query
            The question text; embedded with ``embedder.embed_query``.
        k
            Number of results. Defaults to the constructor's *k*.
        filter
            A :mod:`nanorag.store.filters` mapping. Applied **before** the
            similarity search: the top-*k* is taken over matching chunks
            only. ``None`` or ``{}`` means no filter. A filter that matches
            nothing yields an empty list, not an error.

        Returns
        -------
        list[ScoredChunk]
            At most *k* hits, ``source="dense"``, score descending with ties
            broken by chunk id. Fewer than *k* only when fewer candidates
            exist.

        Raises
        ------
        RetrievalError
            *k* < 1, *filter* is not in the grammar, or a hit's chunk id is
            in the vector store but not in the document store — the two
            have diverged, which the durability ordering (SQLite commits
            first) is meant to make impossible.

        """
        if k is None:
            k = self._k
        if k < 1:
            raise RetrievalError(f"k must be >= 1, got {k}")

        allowed_ids: set[str] | None = None
        if filter:
            allowed_ids = self._docs.filter_chunk_ids(filter)
            if not allowed_ids:
                return []

        query_vector = self._embedder.embed_query(query)
        hits = self._vectors.search(query_vector, k, allowed_ids=allowed_ids)
        if not hits:
            return []

        chunks = self._docs.get_chunks_by_ids([chunk_id for chunk_id, _ in hits])
        results: list[ScoredChunk] = []
        for chunk_id, score in hits:
            chunk = chunks.get(chunk_id)
            if chunk is None:
                raise RetrievalError(
                    "chunk is in the vector store but not the document store",
                    chunk_id=chunk_id,
                )
            results.append(ScoredChunk(chunk=chunk, score=score, source=SOURCE))
        return results
