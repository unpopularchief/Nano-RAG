"""``MultiQueryRetriever`` — retrieve for several phrasings of one question.

plan.md §9 Phase F session F3: "optional query transforms behind flags".
Every other Phase F technique stays inside plan.md §4's one invariant — "the
only network call in the whole engine is the single generation request per
query" — because it works purely from what is already embedded or indexed.
A query transform cannot: turning one question into several better-phrased
ones needs a model, which means a second generation call *before*
retrieval even starts. That is exactly why this is opt-in behind an
explicit ``QueryTransform`` a caller chooses to wire in, never something
``from_defaults()`` or any other retriever composition reaches for on its
own — the "flags" plan.md's own wording calls for.

``QueryTransform`` (``transform(query) -> list[str]``) is the extension
point, written as a ``Protocol`` at its second implementation
(``IdentityQueryTransform``, ``LLMQueryTransform`` — plan.md §15 #1).
``MultiQueryRetriever`` retrieves once per phrasing (the original query
plus whatever the transform adds) against a wrapped retriever and fuses
the result lists with :func:`~nanorag.retrieval.hybrid.rrf_fuse` — the
same RRF arithmetic ``HybridRetriever`` uses across *retrievers*, reused
here across *query phrasings* of one retriever (plan.md §5 rule 1: shared
as one function precisely so the two do not drift apart).
"""

from __future__ import annotations

import logging
from typing import Protocol, runtime_checkable

from nanorag.errors import (
    ConfigError,
    ProviderError,
    QueryTransformError,
    RetrievalError,
)
from nanorag.generation.base import Generator
from nanorag.prompting.templates import Prompt
from nanorag.retrieval.base import Retriever
from nanorag.retrieval.hybrid import DEFAULT_RRF_K, rrf_fuse
from nanorag.store.filters import Filter
from nanorag.types import ScoredChunk

log = logging.getLogger("nanorag")

#: The ``ScoredChunk.source`` a fused (more than one phrasing) result carries.
SOURCE = "query_transform:rrf"

#: Default number of alternative phrasings :class:`LLMQueryTransform` asks for.
DEFAULT_N = 3

QUERY_TRANSFORM_SYSTEM_PROMPT = (
    "You rewrite a user's search question into {n} alternative phrasings "
    "that would help a search engine find the same information — different "
    "wording, synonyms, or a narrower/broader framing. Reply with exactly "
    "{n} lines, one phrasing per line, and nothing else: no numbering, no "
    "quotation marks, no commentary."
)


@runtime_checkable
class QueryTransform(Protocol):
    """Structural protocol: any object with this one method satisfies it."""

    def transform(self, query: str) -> list[str]:
        """Return additional phrasings of *query* to retrieve with.

        The original *query* is always retrieved too — implementations
        return only the *extra* phrasings, never including or repeating
        *query* itself. An empty list means "no expansion."

        Raises
        ------
        QueryTransformError
            A phrasing could not be produced for any reason.

        """
        ...


class IdentityQueryTransform:
    """The default: no expansion, ever fails. ``MultiQueryRetriever``'s no-op."""

    def transform(self, query: str) -> list[str]:
        """Return ``[]`` — *query* is unused."""
        return []


class LLMQueryTransform:
    """Ask a generator for alternative phrasings of a query.

    Parameters
    ----------
    generator
        Any :class:`~nanorag.generation.base.Generator`. This is a second,
        separate model call from the one that eventually answers the
        question — see the module docstring's network-call caveat.
    n
        How many alternative phrasings to request. Defaults to
        :data:`DEFAULT_N`.

    Raises
    ------
    ConfigError
        *n* < 1.

    """

    def __init__(self, generator: Generator, *, n: int = DEFAULT_N) -> None:
        """Store the generator and *n*; validate *n*."""
        if n < 1:
            raise ConfigError(f"n must be >= 1, got {n}")
        self._generator = generator
        self._n = n

    def transform(self, query: str) -> list[str]:
        """Return up to *n* alternative phrasings of *query*, one per line.

        Raises
        ------
        QueryTransformError
            The generator call failed, or its response contained no usable
            lines.

        """
        prompt = Prompt(
            system=QUERY_TRANSFORM_SYSTEM_PROMPT.format(n=self._n),
            user=query,
            # No retrieved (untrusted) content is in play at this stage —
            # the query itself is the caller's own input, nothing to fence
            # against — so a fixed nonce is fine; it exists only to satisfy
            # `Prompt`'s shape.
            nonce="query-transform",
        )
        try:
            generation = self._generator.generate(prompt)
        except ProviderError as exc:
            raise QueryTransformError(
                "query transform generation failed", query=query
            ) from exc
        lines = [line.strip() for line in generation.text.splitlines() if line.strip()]
        if not lines:
            raise QueryTransformError(
                "query transform produced no usable phrasings", query=query
            )
        return lines[: self._n]


class MultiQueryRetriever:
    """Retrieve for the original query plus transform-generated phrasings, RRF-fused.

    Parameters
    ----------
    retriever
        The wrapped :class:`~nanorag.retrieval.base.Retriever`, called once
        per phrasing.
    transform
        Produces additional phrasings. Defaults to
        :class:`IdentityQueryTransform` — with no expansion, exactly one
        phrasing (the original query) is retrieved and this wrapper is a
        pass-through to *retriever*, byte-identical to not wrapping it at
        all.
    k
        Default number of fused results when ``retrieve`` is not given one.
    candidate_k
        How many results *retriever* is asked for per phrasing before
        fusion. Defaults to *k*.
    rrf_k
        The RRF constant, passed to
        :func:`~nanorag.retrieval.hybrid.rrf_fuse`. Defaults to
        :data:`~nanorag.retrieval.hybrid.DEFAULT_RRF_K`.

    Raises
    ------
    RetrievalError
        *k* / *candidate_k* / *rrf_k* < 1.

    """

    def __init__(
        self,
        retriever: Retriever,
        transform: QueryTransform | None = None,
        *,
        k: int = 10,
        candidate_k: int | None = None,
        rrf_k: int = DEFAULT_RRF_K,
    ) -> None:
        """Wire the retriever and transform; validate the parameters."""
        if k < 1:
            raise RetrievalError(f"k must be >= 1, got {k}")
        if candidate_k is not None and candidate_k < 1:
            raise RetrievalError(f"candidate_k must be >= 1, got {candidate_k}")
        if rrf_k < 1:
            raise RetrievalError(f"rrf_k must be >= 1, got {rrf_k}")
        self._retriever = retriever
        self._transform: QueryTransform = (
            transform if transform is not None else IdentityQueryTransform()
        )
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
        """Return the top-*k* chunks fused across every phrasing of *query*.

        A ``transform`` that raises ``QueryTransformError`` degrades to
        retrieving with the original *query* alone (a ``WARNING`` is
        logged) rather than failing the call — the same "an optional stage
        never fails the query" contract ``rerank_hits`` and
        ``Bm25Retriever``'s FTS5 fallback already uphold.

        Returns
        -------
        list[ScoredChunk]
            At most *k* hits. With no expansion (zero extra phrasings),
            these are exactly *retriever*'s own hits for *query*,
            unchanged — ``source`` stays whatever *retriever* set. With one
            or more extra phrasings, RRF-fused across every phrasing's
            result list, ``source="query_transform:rrf"``.

        Raises
        ------
        RetrievalError
            *k* < 1, or the wrapped retriever raises.

        """
        final_k = k if k is not None else self._k
        if final_k < 1:
            raise RetrievalError(f"k must be >= 1, got {final_k}")
        candidate_k = max(self._candidate_k or final_k, final_k)

        try:
            extra = self._transform.transform(query)
        except QueryTransformError:
            log.warning(
                "query transform %r failed, retrieving with the original query only",
                self._transform,
                exc_info=True,
            )
            extra = []

        phrasings = [query, *dict.fromkeys(p for p in extra if p and p != query)]
        if len(phrasings) == 1:
            return self._retriever.retrieve(query, k=final_k, filter=filter)

        result_lists = [
            self._retriever.retrieve(phrasing, k=candidate_k, filter=filter)
            for phrasing in phrasings
        ]
        return rrf_fuse(result_lists, final_k, rrf_k=self._rrf_k, source=SOURCE)
