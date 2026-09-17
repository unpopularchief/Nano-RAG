"""``HybridRetriever`` — combine two or more retrievers by Reciprocal Rank Fusion.

plan.md §9 Phase F session F2: "RRF hybrid fusion". Dense and BM25 rank by
very different, incomparable signals (semantic similarity vs. exact term
overlap) on incomparable score scales, so fusing on raw score would let
whichever retriever's numbers happen to be larger dominate. RRF sidesteps
the scale problem entirely by only looking at *rank*, never score: a
chunk's fused score is ``sum(1 / (rrf_k + rank_r(chunk)) for r in
retrievers)``, where ``rank_r(chunk)`` is its 1-based position in retriever
*r*'s own result list — a chunk absent from *r*'s results contributes
nothing from *r*, not a penalty.
"""

from __future__ import annotations

from nanorag.errors import RetrievalError
from nanorag.retrieval.base import Retriever
from nanorag.store.filters import Filter
from nanorag.types import Chunk, ScoredChunk

#: The ``ScoredChunk.source`` every hit from this retriever carries.
SOURCE = "hybrid:rrf"

#: The standard RRF constant (Cormack, Clarke & Buettcher 2009's own choice,
#: widely reused since without further tuning): large enough that a
#: retriever's very top ranks do not completely dominate the sum, small
#: enough that rank still matters more than mere participation count.
DEFAULT_RRF_K = 60


class HybridRetriever:
    """Fuse two or more retrievers' rankings by Reciprocal Rank Fusion.

    Parameters
    ----------
    *retrievers
        Two or more objects satisfying
        :class:`~nanorag.retrieval.base.Retriever` — most commonly a
        :class:`~nanorag.retrieval.dense.DenseRetriever` and a
        :class:`~nanorag.retrieval.bm25.Bm25Retriever`.
    k
        Default number of fused results when ``retrieve`` is not given one.
    candidate_k
        How many results each retriever is asked for before fusion.
        Defaults to *k* itself; pass a larger value so RRF has a wider pool
        to find overlap in ("retrieve wide, keep narrow", the same idea
        ``Rag.retrieve_k`` applies to reranking — plan.md §9 Phase F).
    rrf_k
        The RRF constant. Defaults to :data:`DEFAULT_RRF_K`.

    Raises
    ------
    RetrievalError
        Fewer than two retrievers are given, or *k* / *candidate_k* /
        *rrf_k* < 1.

    """

    def __init__(
        self,
        *retrievers: Retriever,
        k: int = 10,
        candidate_k: int | None = None,
        rrf_k: int = DEFAULT_RRF_K,
    ) -> None:
        """Wire the retrievers together; validate the parameters."""
        if len(retrievers) < 2:
            raise RetrievalError(
                f"HybridRetriever needs at least 2 retrievers, got {len(retrievers)}"
            )
        if k < 1:
            raise RetrievalError(f"k must be >= 1, got {k}")
        if candidate_k is not None and candidate_k < 1:
            raise RetrievalError(f"candidate_k must be >= 1, got {candidate_k}")
        if rrf_k < 1:
            raise RetrievalError(f"rrf_k must be >= 1, got {rrf_k}")
        self._retrievers = retrievers
        self._k = k
        self._candidate_k = candidate_k
        self._rrf_k = rrf_k

    @property
    def k(self) -> int:
        """Default number of fused results."""
        return self._k

    def retrieve(
        self,
        query: str,
        k: int | None = None,
        filter: Filter | None = None,
    ) -> list[ScoredChunk]:
        """Return the top-*k* chunks for *query*, fused by RRF.

        Every retriever is asked for ``max(candidate_k, k)`` results with
        the same *filter*, then fused: a chunk's score is the sum of
        ``1 / (rrf_k + rank)`` over every retriever whose result list
        contains it, ``rank`` 1-based within that list. Ties broken by
        chunk id, matching every other retriever's convention.

        Returns
        -------
        list[ScoredChunk]
            At most *k* hits, ``source="hybrid:rrf"``. Each unique chunk's
            ``Chunk`` object is taken from whichever retriever returned it
            first — their content for a given ``chunk_id`` is always
            identical, since every retriever reads the same document store.

        Raises
        ------
        RetrievalError
            *k* < 1, or a wrapped retriever raises (a bad *filter*, its own
            construction invariant, …).

        """
        final_k = k if k is not None else self._k
        if final_k < 1:
            raise RetrievalError(f"k must be >= 1, got {final_k}")
        candidate_k = max(self._candidate_k or final_k, final_k)

        fused: dict[str, float] = {}
        chunk_for_id: dict[str, Chunk] = {}
        for retriever in self._retrievers:
            hits = retriever.retrieve(query, k=candidate_k, filter=filter)
            for rank, hit in enumerate(hits, start=1):
                cid = hit.chunk.chunk_id
                fused[cid] = fused.get(cid, 0.0) + 1.0 / (self._rrf_k + rank)
                chunk_for_id.setdefault(cid, hit.chunk)

        ranked = sorted(fused.items(), key=lambda pair: (-pair[1], pair[0]))[:final_k]
        return [
            ScoredChunk(chunk=chunk_for_id[cid], score=score, source=SOURCE)
            for cid, score in ranked
        ]
