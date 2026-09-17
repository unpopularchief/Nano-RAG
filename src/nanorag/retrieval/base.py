"""The ``Retriever`` protocol — query -> ranked ``ScoredChunk``s.

Written now, at the second implementation (plan.md §15 #1): ``DenseRetriever``
(Phase C) and ``Bm25Retriever`` / ``HybridRetriever`` (Phase F session F2)
all satisfy this one shape, so ``Rag`` — and anything composing retrievers by
hand, per plan.md §7 — can hold any of them behind the same attribute.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from nanorag.store.filters import Filter
from nanorag.types import ScoredChunk


@runtime_checkable
class Retriever(Protocol):
    """Structural protocol: any object with this shape satisfies it."""

    @property
    def k(self) -> int:
        """Default number of results when ``retrieve`` is not given one.

        Every implementation (``DenseRetriever``, ``Bm25Retriever``,
        ``HybridRetriever``) carries this, and the evaluation runner reads
        it to report what a sweep row actually ran with (``nanorag.
        evaluation.runner.evaluate_answers``'s config).
        """
        ...

    def retrieve(
        self, query: str, k: int | None = None, filter: Filter | None = None
    ) -> list[ScoredChunk]:
        """Return the top-*k* chunks for *query*, highest score first.

        Parameters
        ----------
        query
            The question text.
        k
            Number of results. ``None`` defers to the retriever's own
            default.
        filter
            A :mod:`nanorag.store.filters` mapping, applied as a
            **pre**-filter — the top-*k* is taken over matching chunks only
            (plan.md §15 #7).

        Raises
        ------
        RetrievalError
            *k* < 1, or *filter* is not in the grammar.

        """
        ...
