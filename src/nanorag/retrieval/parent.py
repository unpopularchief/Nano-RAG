"""``ParentExpandingRetriever`` — widen each hit to its surrounding neighbours.

plan.md §9 Phase F session F3: "parent-document / neighbour expansion".
Small chunks retrieve precisely (a narrow embedding matches a narrow span
closely) but can starve the generator of the surrounding sentence that
would have made the answer unambiguous — the classic "small-to-big"
retrieval trade-off. This retriever keeps the wrapped retriever's ranking
exactly as-is (it changes *what a hit contains*, never *which hits or in
what order*) and, for each one, replaces its chunk with the union of the
document's chunks within ``window`` ordinals either side, re-sliced
straight from the owning ``Document.text`` — never a concatenation of the
narrow chunks' own ``.text`` fields, which would double up whatever
``overlap_tokens`` the chunker used between them.

**Composition order matters, the mirror image of** ``retrieval.mmr``'s
note: apply this as the *outermost* wrapper — around
``MmrRetriever``/``HybridRetriever``/a reranked pool — never innermost,
since everything it wraps needs the original, narrow chunk ids and
embeddings to do its own job (MMR's diversity lookups, a reranker's
narrower-is-more-precise scoring) before expansion ever touches them.

**Two documented limitations, not fixed here.** ``Chunk.token_count`` on an
expanded chunk is the *sum* of its window members' counts, which
double-counts whatever tokens the chunker's own ``overlap_tokens`` put in
both of two adjacent chunks — an overestimate, informational only (nothing
downstream trusts it: ``ContextBuilder`` always recounts the real prompt
text with its own counter). And two hits that land next to each other in
the *same* result list (ordinals 5 and 6, say, both surviving to the final
top-*k*) expand into overlapping-but-not-identical spans that
``ContextBuilder``'s exact-text dedup will not catch, wasting some of the
token budget on repeated sentences — a genuine cost, left as a known
trade-off rather than solved by a window-merging pass across hits.
"""

from __future__ import annotations

from nanorag.errors import RetrievalError
from nanorag.hashing import chunk_id as compute_chunk_id
from nanorag.retrieval.base import Retriever
from nanorag.store.filters import Filter
from nanorag.store.sqlite_docs import SqliteDocumentStore
from nanorag.types import Chunk, ScoredChunk


class ParentExpandingRetriever:
    """Expand a wrapped retriever's hits to their neighbouring chunks.

    Parameters
    ----------
    retriever
        The wrapped :class:`~nanorag.retrieval.base.Retriever`. Its
        ranking and hit count are preserved exactly; only each hit's
        ``Chunk`` is replaced.
    docs
        The document store a hit's neighbours and owning document are read
        from.
    window
        How many chunks either side of a hit (by ``ordinal``) to fold in.
        ``1`` (the default) means "one chunk before, the hit itself, one
        chunk after."
    k
        Default number of results when ``retrieve`` is not given one.

    Raises
    ------
    RetrievalError
        *window* < 1, or *k* < 1.

    """

    def __init__(
        self,
        retriever: Retriever,
        docs: SqliteDocumentStore,
        *,
        window: int = 1,
        k: int = 10,
    ) -> None:
        """Wire the wrapped retriever and document store; validate the parameters."""
        if window < 1:
            raise RetrievalError(f"window must be >= 1, got {window}")
        if k < 1:
            raise RetrievalError(f"k must be >= 1, got {k}")
        self._retriever = retriever
        self._docs = docs
        self._window = window
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
        """Return the wrapped retriever's hits, each widened to its neighbours.

        Returns
        -------
        list[ScoredChunk]
            Exactly as many hits, in exactly the same order and with the
            same ``score``/``source``, as the wrapped retriever returned —
            only ``chunk`` differs on a hit whose neighbours exist. A hit
            at a document's first or last ordinal, or in a single-chunk
            document, is returned unchanged (there is nothing to fold in
            on that side, or either side).

        Raises
        ------
        RetrievalError
            *k* < 1, or the wrapped retriever raises.

        """
        if k is None:
            k = self._k
        if k < 1:
            raise RetrievalError(f"k must be >= 1, got {k}")

        hits = self._retriever.retrieve(query, k=k, filter=filter)
        return [self._expand(hit) for hit in hits]

    def _expand(self, hit: ScoredChunk) -> ScoredChunk:
        """Return *hit* with its chunk widened to its ``window``-neighbours."""
        anchor = hit.chunk
        by_ordinal = {c.ordinal: c for c in self._docs.get_chunks(anchor.doc_id)}
        if anchor.ordinal not in by_ordinal:
            return hit  # store/candidate diverged (shouldn't happen); leave as-is

        lo, hi = anchor.ordinal - self._window, anchor.ordinal + self._window
        window_chunks = [by_ordinal[o] for o in range(lo, hi + 1) if o in by_ordinal]
        if len(window_chunks) <= 1:
            return hit  # nothing to fold in (a document edge, or a lone chunk)

        document = self._docs.get_document(anchor.doc_id)
        if document is None:
            return hit  # same defensive case as above

        start = min(c.start_char for c in window_chunks)
        end = max(c.end_char for c in window_chunks)
        text = document.text[start:end]
        expanded = Chunk(
            chunk_id=compute_chunk_id(anchor.doc_id, anchor.ordinal, text),
            doc_id=anchor.doc_id,
            ordinal=anchor.ordinal,
            text=text,
            start_char=start,
            end_char=end,
            token_count=sum(c.token_count for c in window_chunks),
            metadata=anchor.metadata,
        )
        return ScoredChunk(chunk=expanded, score=hit.score, source=hit.source)
