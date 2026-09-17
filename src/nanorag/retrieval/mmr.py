"""``MmrRetriever`` — diversify a candidate pool by Maximal Marginal Relevance.

plan.md §9 Phase F session F3: "MMR diversity". A plain top-*k* can return
*k* near-duplicates of the single most relevant passage (three chunks that
all restate the same sentence, say) while a genuinely different, still
relevant angle on the question sits just outside the cut. MMR selects
greedily, trading a candidate's relevance against how similar it is to what
has already been picked (Carbonell & Goldstein 1998):

    MMR(d) = lambda_mult * relevance(d) - (1 - lambda_mult) * max(sim(d, s)
             for s in selected)

Relevance comes from whichever retriever/reranker produced the candidate
pool — MMR does not re-score relevance itself, only diversity — normalised
to ``[0, 1]`` within the pool so ``lambda_mult`` means the same thing
regardless of the wrapped stage's own score scale (cosine, BM25, RRF, ...).
Diversity is always semantic: the pairwise term comes from the chunks' own
embeddings in the vector store, fetched by id
(:meth:`~nanorag.store.numpy_store.NumpyVectorStore.get_vectors`), never
recomputed — every chunk is embedded at ingest regardless of which
retriever finds it later.

**Composition order matters.** ``MmrRetriever`` looks candidates' vectors
up by their *current* ``chunk_id`` — wrap it *inside* (closer to the base
retriever than) anything that changes chunk identity, most notably
``retrieval.parent.ParentExpandingRetriever``, never outside it. Wrapping
another :class:`~nanorag.retrieval.base.Retriever` (dense, BM25, hybrid, a
reranked pool, ...) is otherwise exactly the established pattern.
"""

from __future__ import annotations

import numpy as np

from nanorag.errors import RetrievalError
from nanorag.retrieval.base import Retriever
from nanorag.store.filters import Filter
from nanorag.store.numpy_store import NumpyVectorStore
from nanorag.types import ScoredChunk

#: The ``ScoredChunk.source`` every hit from this retriever carries.
SOURCE = "mmr"

#: Balanced default: relevance and diversity weighted equally.
DEFAULT_LAMBDA_MULT = 0.5


class MmrRetriever:
    """Re-select a wrapped retriever's candidates for relevance *and* diversity.

    Parameters
    ----------
    retriever
        The wrapped :class:`~nanorag.retrieval.base.Retriever` providing
        both the candidate pool and each candidate's relevance signal (its
        own ``ScoredChunk.score``).
    vectors
        Where each candidate's embedding is looked up by chunk id, for the
        diversity term. Must be the same index *retriever* (transitively)
        reads from.
    k
        Default number of results when ``retrieve`` is not given one.
    candidate_k
        How many results to pull from *retriever* before MMR selects *k*
        from them. Defaults to *k*; a wider pool gives MMR more to choose
        from ("retrieve wide, keep narrow", plan.md §9 Phase F).
    lambda_mult
        Relevance/diversity trade-off in ``[0, 1]``. ``1.0`` ignores
        diversity entirely (MMR degenerates to the wrapped retriever's own
        order); ``0.0`` ignores relevance entirely and picks the most
        mutually-dissimilar set. Defaults to :data:`DEFAULT_LAMBDA_MULT`.

    Raises
    ------
    RetrievalError
        *k* / *candidate_k* < 1, or *lambda_mult* is not in ``[0, 1]``.

    """

    def __init__(
        self,
        retriever: Retriever,
        vectors: NumpyVectorStore,
        *,
        k: int = 10,
        candidate_k: int | None = None,
        lambda_mult: float = DEFAULT_LAMBDA_MULT,
    ) -> None:
        """Wire the wrapped retriever and vector store; validate the parameters."""
        if k < 1:
            raise RetrievalError(f"k must be >= 1, got {k}")
        if candidate_k is not None and candidate_k < 1:
            raise RetrievalError(f"candidate_k must be >= 1, got {candidate_k}")
        if not 0.0 <= lambda_mult <= 1.0:
            raise RetrievalError(f"lambda_mult must be in [0, 1], got {lambda_mult}")
        self._retriever = retriever
        self._vectors = vectors
        self._k = k
        self._candidate_k = candidate_k
        self._lambda_mult = lambda_mult

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
        """Return the top-*k* chunks for *query*, selected by MMR.

        Returns
        -------
        list[ScoredChunk]
            At most *k* hits, ``source="mmr"``, in MMR selection order
            (first pick most relevant, then trading relevance for
            diversity). ``score`` is each pick's MMR score at the moment it
            was selected (``lambda_mult * relevance - (1 - lambda_mult) *
            worst-case similarity to what's already picked``) — informative
            for debugging a selection, not comparable across queries or
            candidate pools the way a retriever's own score is. A candidate
            with no embedding in *vectors* (should not happen — every chunk
            is embedded at ingest — but not assumed) is treated as
            maximally diverse from everything already selected, never
            excluded.

        Raises
        ------
        RetrievalError
            *k* < 1, or the wrapped retriever raises.

        """
        final_k = k if k is not None else self._k
        if final_k < 1:
            raise RetrievalError(f"k must be >= 1, got {final_k}")
        candidate_k = max(self._candidate_k or final_k, final_k)

        candidates = self._retriever.retrieve(query, k=candidate_k, filter=filter)
        if not candidates:
            return []

        candidate_vectors = self._vectors.get_vectors(
            [hit.chunk.chunk_id for hit in candidates]
        )
        relevance = _normalize(np.array([hit.score for hit in candidates]))

        selected: list[int] = []
        selected_scores: list[float] = []
        remaining = list(range(len(candidates)))
        while remaining and len(selected) < final_k:
            best_pos, best_score = _pick_next(
                candidates,
                candidate_vectors,
                relevance,
                selected,
                remaining,
                self._lambda_mult,
            )
            selected.append(best_pos)
            selected_scores.append(best_score)
            remaining.remove(best_pos)

        return [
            ScoredChunk(chunk=candidates[i].chunk, score=score, source=SOURCE)
            for i, score in zip(selected, selected_scores, strict=True)
        ]


def _normalize(scores: np.ndarray) -> np.ndarray:
    """Min-max scale *scores* to ``[0, 1]``; a flat pool maps to all-``1.0``."""
    lo, hi = float(scores.min()), float(scores.max())
    if hi <= lo:
        return np.ones_like(scores)
    return (scores - lo) / (hi - lo)


def _pick_next(
    candidates: list[ScoredChunk],
    candidate_vectors: dict[str, np.ndarray],
    relevance: np.ndarray,
    selected: list[int],
    remaining: list[int],
    lambda_mult: float,
) -> tuple[int, float]:
    """Return ``(index, mmr_score)`` of the best pick in *remaining*.

    Ties are broken by chunk id.
    """
    best_pos: int | None = None
    best_score = float("-inf")
    best_id = ""
    for i in remaining:
        vector = candidate_vectors.get(candidates[i].chunk.chunk_id)
        if vector is None or not selected:
            diversity_penalty = 0.0
        else:
            diversity_penalty = max(
                (
                    float(vector @ other)
                    for j in selected
                    if (other := candidate_vectors.get(candidates[j].chunk.chunk_id))
                    is not None
                ),
                default=0.0,
            )
        mmr_score = (
            lambda_mult * float(relevance[i]) - (1 - lambda_mult) * diversity_penalty
        )
        cid = candidates[i].chunk.chunk_id
        if (
            best_pos is None
            or mmr_score > best_score
            or (mmr_score == best_score and cid < best_id)
        ):
            best_pos, best_score, best_id = i, mmr_score, cid
    assert best_pos is not None  # `remaining` is never empty when called
    return best_pos, best_score
