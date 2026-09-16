"""The ``Reranker`` protocol — re-score a candidate list, per plan.md §6.

A reranker's job, precisely: given a query and the ``ScoredChunk``s a
retriever already found, return the best ``top_n`` of them, re-scored and
reordered. Explicitly not its job: fetching more candidates — that is what
``retrieve_k`` is for (plan.md §9 Phase F "retrieve-k / rerank-to-n
wiring"): the retriever over-fetches a wider candidate pool, the reranker
narrows it back down.

``IdentityReranker`` (the default — plan.md §6) and ``JinaReranker`` are the
two implementations this protocol is written at (plan.md §15 #1);
``LocalCrossEncoderReranker`` is a third. A reranker that cannot re-score —
a network failure, a missing model, a malformed response — raises
``RerankError``; ``Rag.retrieve()`` catches exactly that and falls back to
the retriever's own order with a warning, never failing the query (plan.md
§9 Phase F Tests).
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from nanorag.types import ScoredChunk


@runtime_checkable
class Reranker(Protocol):
    """Structural protocol: any object with this one method satisfies it."""

    def rerank(
        self, query: str, hits: list[ScoredChunk], top_n: int
    ) -> list[ScoredChunk]:
        """Return the best ``top_n`` of *hits*, re-scored for *query*.

        Parameters
        ----------
        query
            The original question text.
        hits
            Candidates already found by a retriever, most-similar first.
            May contain more than *top_n* items — that oversampling is the
            point (plan.md §9 Phase F).
        top_n
            How many to return.

        Returns
        -------
        list[ScoredChunk]
            At most ``top_n`` hits. Implementations decide their own
            ``ScoredChunk.source`` (plan.md §5 core types: ``"dense"`` |
            ``"bm25"`` | ``"rerank:<name>"``).

        Raises
        ------
        RerankError
            Re-scoring failed for any reason.

        """
        ...
