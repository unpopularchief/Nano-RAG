"""``Bm25Retriever`` — lexical BM25 search, SQLite FTS5 by default.

plan.md §9 Phase F session F2: "BM25 over SQLite FTS5 (zero deps, capability
check + NumPy fallback)". FTS5 ships in SQLite itself — no new dependency —
but is an optional compile-time feature some minimal SQLite builds omit.
``SqliteDocumentStore`` checks for it once, at open time
(:attr:`~nanorag.store.sqlite_docs.SqliteDocumentStore.fts5_available`), and
keeps a ``chunks_fts`` virtual table synced with ``chunks`` via triggers, so
this retriever never has to maintain a text index itself when FTS5 is
present — it only has to turn *query* into an FTS5 ``MATCH`` expression
(:func:`_fts5_match_query`) and hand it to
:meth:`~nanorag.store.sqlite_docs.SqliteDocumentStore.search_bm25`.

When FTS5 is not present, :class:`Bm25Retriever` falls back to
:func:`numpy_bm25_scores`: a small in-memory Okapi BM25 (the same ``k1=1.2``,
``b=0.75`` SQLite's own ``bm25()`` uses, so it is not just *a* BM25, but the
*same* BM25 the FTS5 path would compute) over every candidate chunk's text,
recomputed from scratch on every call — correct, not indexed, and only ever
exercised when FTS5 is missing.
"""

from __future__ import annotations

import re

import numpy as np

from nanorag.errors import RetrievalError
from nanorag.hashing import normalize_text
from nanorag.store.filters import Filter
from nanorag.store.sqlite_docs import SqliteDocumentStore
from nanorag.types import Chunk, ScoredChunk

#: The ``ScoredChunk.source`` every hit from this retriever carries.
SOURCE = "bm25"

#: Word tokenizer shared by the FTS5 MATCH-query builder and the NumPy
#: fallback, so both paths treat text the same way: words, case-folded,
#: punctuation stripped — close enough to FTS5's default ``unicode61``
#: tokenizer for BM25 purposes, and the same normalisation
#: ``tests.fakes.FakeEmbedder`` already uses.
_TOKEN_RE = re.compile(r"\w+")

#: Okapi BM25 constants, matching SQLite FTS5's ``bm25()`` defaults exactly
#: (https://sqlite.org/fts5.html#the_bm25_function) so the NumPy fallback
#: computes the same ranking formula the FTS5 path would, not merely a
#: similar one.
_K1 = 1.2
_B = 0.75


def _tokenize(text: str) -> list[str]:
    """Lowercase, normalise and split *text* into word tokens."""
    return _TOKEN_RE.findall(normalize_text(text).lower())


def _fts5_match_query(query: str) -> str | None:
    """Build an FTS5 ``MATCH`` expression that ORs every query term.

    ``None`` if *query* has no tokens (empty, or all punctuation) — the
    caller must not pass an empty string to FTS5's ``MATCH``, which is a
    syntax error, not zero results.

    Every term is tokenized the same way the index was built
    (:func:`_tokenize`) and phrase-quoted, so a token can never be
    misinterpreted as FTS5 query syntax (``AND``, ``NOT``, ``*``, …) — it is
    always a literal word match. Terms are ORed rather than the FTS5
    default AND: a chunk containing *any* query term is a BM25 candidate,
    ranked by how many terms and how rare they are, not required to contain
    every one.
    """
    terms = dict.fromkeys(_tokenize(query))  # de-dup, keep first-seen order
    if not terms:
        return None
    return " OR ".join(f'"{t}"' for t in terms)


def numpy_bm25_scores(query: str, chunks: list[Chunk]) -> dict[str, float]:
    """Score *chunks* against *query* with an in-memory Okapi BM25.

    The FTS5-unavailable fallback (plan.md §9 Phase F session F2 "NumPy
    fallback"): correct, but recomputed from scratch on every call, since
    there is nowhere durable to keep an index without FTS5. Only chunks
    sharing at least one query term score above ``0.0``; the rest are
    absent from the result rather than included at a floor score.
    """
    query_terms = _tokenize(query)
    if not query_terms or not chunks:
        return {}

    doc_tokens = [_tokenize(c.text) for c in chunks]
    doc_lengths = np.array([len(toks) for toks in doc_tokens], dtype=np.float64)
    n_docs = len(chunks)
    avg_len = float(doc_lengths.mean())
    if avg_len == 0.0:
        return {}  # every candidate is empty text; nothing can match a term

    scores = np.zeros(n_docs, dtype=np.float64)
    for term in dict.fromkeys(query_terms):
        term_freqs = np.array(
            [toks.count(term) for toks in doc_tokens], dtype=np.float64
        )
        df = int((term_freqs > 0).sum())
        if df == 0:
            continue
        idf = np.log((n_docs - df + 0.5) / (df + 0.5) + 1.0)
        denom = term_freqs + _K1 * (1 - _B + _B * doc_lengths / avg_len)
        scores += idf * (term_freqs * (_K1 + 1)) / denom

    return {
        c.chunk_id: float(s) for c, s in zip(chunks, scores, strict=True) if s > 0.0
    }


class Bm25Retriever:
    """Lexical BM25 top-*k*, via SQLite FTS5 when available.

    Parameters
    ----------
    docs
        The document store. Its ``fts5_available`` decides which of the two
        scoring paths every call takes; both satisfy
        :class:`~nanorag.retrieval.base.Retriever` and return
        ``ScoredChunk``s with ``source="bm25"``.
    k
        Default number of results when ``retrieve`` is not given one.

    Raises
    ------
    RetrievalError
        *k* < 1.

    """

    def __init__(self, docs: SqliteDocumentStore, *, k: int = 10) -> None:
        """Wire the document store; validate *k*."""
        if k < 1:
            raise RetrievalError(f"k must be >= 1, got {k}")
        self._docs = docs
        self._k = k

    @property
    def k(self) -> int:
        """Default number of results."""
        return self._k

    @property
    def fts5_available(self) -> bool:
        """Whether this retriever is using FTS5 (``False`` -> the NumPy fallback)."""
        return self._docs.fts5_available

    def retrieve(
        self,
        query: str,
        k: int | None = None,
        filter: Filter | None = None,
    ) -> list[ScoredChunk]:
        """Return the top-*k* chunks for *query* by BM25 score, highest first.

        Parameters mirror
        :meth:`~nanorag.retrieval.dense.DenseRetriever.retrieve` exactly,
        including the pre-filter semantics: *filter* restricts the
        candidate set before the top-*k* cut, never after.

        Returns
        -------
        list[ScoredChunk]
            At most *k* hits, ``source="bm25"``. A query with no recognised
            terms, or that matches nothing, returns ``[]`` — never an error.

        Raises
        ------
        RetrievalError
            *k* < 1, *filter* is not in the grammar, or a hit's chunk id is
            in the BM25 index but not the document store.

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

        if self._docs.fts5_available:
            match_query = _fts5_match_query(query)
            if match_query is None:
                return []
            hits = self._docs.search_bm25(match_query, k, allowed_ids=allowed_ids)
        else:
            candidates = list(
                self._docs.get_chunks_by_ids(list(allowed_ids)).values()
                if allowed_ids is not None
                else self._docs.iter_chunks()
            )
            scores = numpy_bm25_scores(query, candidates)
            ranked = sorted(scores.items(), key=lambda pair: (-pair[1], pair[0]))
            hits = ranked[:k]

        if not hits:
            return []
        chunks = self._docs.get_chunks_by_ids([chunk_id for chunk_id, _ in hits])
        results: list[ScoredChunk] = []
        for chunk_id, score in hits:
            chunk = chunks.get(chunk_id)
            if chunk is None:
                raise RetrievalError(
                    "chunk is in the BM25 index but not the document store",
                    chunk_id=chunk_id,
                )
            results.append(ScoredChunk(chunk=chunk, score=score, source=SOURCE))
        return results
