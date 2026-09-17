"""``SqliteDocumentStore`` — durable documents, chunks and embeddings.

Owns four tables: ``documents`` and ``chunks`` (the document registry, per
plan.md §6 — "durable text + metadata + document registry", not similarity
math), ``embeddings`` (the vectors, the durable source of truth a
``VectorStore`` is rebuilt from on restart), and ``meta`` (the ``model_id`` /
``dim`` the index was built with). ``upsert_document`` writes a document and
replaces its chunks in one transaction; a re-ingest's stale chunks — and,
via ``ON DELETE CASCADE``, their embeddings — are removed rather than left
behind. A failure partway through either write leaves no row changed.

A fifth, optional table, ``chunks_fts`` (plan.md §9 Phase F session F2:
"BM25 over SQLite FTS5 ... capability check"), mirrors ``chunks`` for
lexical search. FTS5 ships in SQLite itself — no new dependency — but is an
optional compile-time feature some minimal SQLite builds omit, so it is
never assumed: :meth:`_init_fts5` tries to create it and records whether
that worked as :attr:`fts5_available`. When it did, ``AFTER INSERT`` /
``AFTER DELETE`` triggers on ``chunks`` keep it in sync automatically —
including through the ``ON DELETE CASCADE`` a document delete triggers,
which is why ``PRAGMA recursive_triggers`` is turned on below (SQLite does
not fire a child table's own triggers for a foreign-key-cascaded delete
otherwise) — so no code in :meth:`upsert_document` / :meth:`delete_document`
has to know ``chunks_fts`` exists at all.
"""

from __future__ import annotations

import json
import sqlite3
from collections.abc import Collection, Iterator, Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np

from nanorag.errors import IndexModelMismatch, StoreError
from nanorag.store.filters import Filter, compile_filter
from nanorag.types import Chunk, Document

_SCHEMA = """
CREATE TABLE IF NOT EXISTS documents (
    doc_id TEXT PRIMARY KEY,
    source_uri TEXT NOT NULL UNIQUE,
    text TEXT NOT NULL,
    content_hash TEXT NOT NULL,
    mime TEXT,
    metadata TEXT NOT NULL,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS chunks (
    chunk_id TEXT PRIMARY KEY,
    doc_id TEXT NOT NULL REFERENCES documents(doc_id) ON DELETE CASCADE,
    ordinal INTEGER NOT NULL,
    text TEXT NOT NULL,
    start_char INTEGER NOT NULL,
    end_char INTEGER NOT NULL,
    token_count INTEGER NOT NULL,
    metadata TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_chunks_doc_id ON chunks(doc_id);

CREATE TABLE IF NOT EXISTS embeddings (
    chunk_id TEXT PRIMARY KEY REFERENCES chunks(chunk_id) ON DELETE CASCADE,
    model_id TEXT NOT NULL,
    dim INTEGER NOT NULL,
    vector BLOB NOT NULL
);

CREATE TABLE IF NOT EXISTS meta (
    key TEXT PRIMARY KEY,
    value TEXT NOT NULL
);
"""

_FTS_SCHEMA = """
CREATE VIRTUAL TABLE IF NOT EXISTS chunks_fts USING fts5(chunk_id UNINDEXED, text);

CREATE TRIGGER IF NOT EXISTS chunks_fts_ai AFTER INSERT ON chunks BEGIN
    INSERT INTO chunks_fts(chunk_id, text) VALUES (new.chunk_id, new.text);
END;

CREATE TRIGGER IF NOT EXISTS chunks_fts_ad AFTER DELETE ON chunks BEGIN
    DELETE FROM chunks_fts WHERE chunk_id = old.chunk_id;
END;
"""

_MODEL_ID_KEY = "embedding_model_id"
_DIM_KEY = "embedding_dim"

#: Chunk ids per ``IN (...)`` query in :meth:`SqliteDocumentStore.get_chunks_by_ids`.
_IN_BATCH = 500


def _now() -> str:
    """Return the current UTC time as an ISO-8601 string."""
    return datetime.now(UTC).isoformat()


def _row_to_document(row: tuple[Any, ...]) -> Document:
    """Build a ``Document`` from a ``documents`` row (see the SELECTs below)."""
    doc_id, source_uri, text, content_hash, metadata_json = row
    return Document(
        doc_id=doc_id,
        source_uri=source_uri,
        text=text,
        content_hash=content_hash,
        metadata=json.loads(metadata_json),
    )


def _row_to_chunk(row: tuple[Any, ...]) -> Chunk:
    """Build a ``Chunk`` from a ``chunks`` row (see the SELECTs below)."""
    (
        chunk_id,
        doc_id,
        ordinal,
        text,
        start_char,
        end_char,
        token_count,
        metadata_json,
    ) = row
    return Chunk(
        chunk_id=chunk_id,
        doc_id=doc_id,
        ordinal=ordinal,
        text=text,
        start_char=start_char,
        end_char=end_char,
        token_count=token_count,
        metadata=json.loads(metadata_json),
    )


class SqliteDocumentStore:
    """SQLite-backed durable store for documents, chunks and embeddings.

    Parameters
    ----------
    path
        Path to the SQLite database file. Created (with its schema) if it
        does not already exist.

    """

    def __init__(self, path: str | Path) -> None:
        """Open (creating if needed) the database at *path* and its schema."""
        self._path = Path(path)
        self._conn = sqlite3.connect(str(self._path))
        self._conn.execute("PRAGMA foreign_keys = ON")
        self._conn.execute("PRAGMA recursive_triggers = ON")
        self._conn.executescript(_SCHEMA)
        self._conn.commit()
        self._fts5_available = self._init_fts5()

    def _init_fts5(self) -> bool:
        """Create (or detect) the FTS5 index; return whether FTS5 is available.

        A capability check, not an assumption (plan.md §9 Phase F session
        F2): some SQLite builds omit the optional FTS5 extension, caught
        here as ``sqlite3.OperationalError`` rather than propagated.
        ``chunks_fts`` and its sync triggers are ``IF NOT EXISTS``, so
        reopening an existing store is a no-op; the backfill below then
        covers the one case the triggers alone cannot — ``chunks`` rows
        written before FTS5 became available (or before this table
        existed) on this machine.
        """
        try:
            self._conn.executescript(_FTS_SCHEMA)
        except sqlite3.OperationalError:
            return False
        with self._conn:
            self._conn.execute(
                "INSERT INTO chunks_fts(chunk_id, text) "
                "SELECT chunk_id, text FROM chunks "
                "WHERE chunk_id NOT IN (SELECT chunk_id FROM chunks_fts)"
            )
        return True

    @property
    def fts5_available(self) -> bool:
        """Whether this store's SQLite build has the FTS5 extension compiled in.

        ``False`` means :meth:`search_bm25` cannot be used — callers (see
        ``nanorag.retrieval.bm25.Bm25Retriever``) fall back to a NumPy BM25
        computed from :meth:`iter_chunks` instead.
        """
        return self._fts5_available

    def close(self) -> None:
        """Close the underlying SQLite connection."""
        self._conn.close()

    def __enter__(self) -> SqliteDocumentStore:
        """Return self, for use as a context manager."""
        return self

    def __exit__(self, *exc_info: object) -> None:
        """Close the connection on exiting the ``with`` block."""
        self.close()

    # -- documents + chunks ------------------------------------------------

    def upsert_document(self, document: Document, chunks: Sequence[Chunk]) -> None:
        """Write *document* and replace its chunks, transactionally.

        Parameters
        ----------
        document
            The document to store (inserted, or updated if ``doc_id`` exists;
            ``created_at`` is preserved across an update).
        chunks
            The document's current chunks. Any chunk previously stored for
            this ``doc_id`` and not in *chunks* is removed, and so is any
            embedding stored for it (``ON DELETE CASCADE``).

        Raises
        ------
        StoreError
            A chunk's ``doc_id`` does not match ``document.doc_id``.

        """
        for chunk in chunks:
            if chunk.doc_id != document.doc_id:
                raise StoreError(
                    "chunk.doc_id does not match document.doc_id",
                    chunk_id=chunk.chunk_id,
                    expected=document.doc_id,
                    got=chunk.doc_id,
                )

        now = _now()
        with self._conn:
            self._conn.execute(
                """
                INSERT INTO documents
                    (doc_id, source_uri, text, content_hash, mime, metadata,
                     created_at, updated_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(doc_id) DO UPDATE SET
                    source_uri = excluded.source_uri,
                    text = excluded.text,
                    content_hash = excluded.content_hash,
                    mime = excluded.mime,
                    metadata = excluded.metadata,
                    updated_at = excluded.updated_at
                """,
                (
                    document.doc_id,
                    document.source_uri,
                    document.text,
                    document.content_hash,
                    document.metadata.get("nanorag.mime"),
                    json.dumps(dict(document.metadata)),
                    now,
                    now,
                ),
            )
            self._conn.execute(
                "DELETE FROM chunks WHERE doc_id = ?", (document.doc_id,)
            )
            self._conn.executemany(
                """
                INSERT INTO chunks
                    (chunk_id, doc_id, ordinal, text, start_char, end_char,
                     token_count, metadata)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                [
                    (
                        c.chunk_id,
                        c.doc_id,
                        c.ordinal,
                        c.text,
                        c.start_char,
                        c.end_char,
                        c.token_count,
                        json.dumps(dict(c.metadata)),
                    )
                    for c in chunks
                ],
            )

    def get_document(self, doc_id: str) -> Document | None:
        """Return the document for *doc_id*, or ``None`` if it is not stored."""
        row = self._conn.execute(
            "SELECT doc_id, source_uri, text, content_hash, metadata "
            "FROM documents WHERE doc_id = ?",
            (doc_id,),
        ).fetchone()
        return _row_to_document(row) if row is not None else None

    def iter_documents(self) -> Iterator[Document]:
        """Yield every stored document, ordered by ``doc_id``."""
        cursor = self._conn.execute(
            "SELECT doc_id, source_uri, text, content_hash, metadata "
            "FROM documents ORDER BY doc_id"
        )
        for row in cursor:
            yield _row_to_document(row)

    def count_documents(self) -> int:
        """Return the number of stored documents."""
        row = self._conn.execute("SELECT COUNT(*) FROM documents").fetchone()
        return int(row[0])

    def count_chunks(self, doc_id: str | None = None) -> int:
        """Return the number of stored chunks, for *doc_id* or in total."""
        if doc_id is None:
            row = self._conn.execute("SELECT COUNT(*) FROM chunks").fetchone()
        else:
            row = self._conn.execute(
                "SELECT COUNT(*) FROM chunks WHERE doc_id = ?", (doc_id,)
            ).fetchone()
        return int(row[0])

    def delete_document(self, doc_id: str) -> None:
        """Delete a document and, via cascade, its chunks and embeddings."""
        with self._conn:
            self._conn.execute("DELETE FROM documents WHERE doc_id = ?", (doc_id,))

    def get_chunks(self, doc_id: str) -> list[Chunk]:
        """Return *doc_id*'s chunks, in ordinal order."""
        cursor = self._conn.execute(
            "SELECT chunk_id, doc_id, ordinal, text, start_char, end_char, "
            "token_count, metadata FROM chunks WHERE doc_id = ? ORDER BY ordinal",
            (doc_id,),
        )
        return [_row_to_chunk(row) for row in cursor]

    def iter_chunks(self) -> Iterator[Chunk]:
        """Yield every stored chunk, ordered by document then ordinal."""
        cursor = self._conn.execute(
            "SELECT chunk_id, doc_id, ordinal, text, start_char, end_char, "
            "token_count, metadata FROM chunks ORDER BY doc_id, ordinal"
        )
        for row in cursor:
            yield _row_to_chunk(row)

    def get_chunks_by_ids(self, chunk_ids: Sequence[str]) -> dict[str, Chunk]:
        """Return the stored chunks among *chunk_ids*, keyed by id.

        Ids that are not stored are simply absent from the result. This is
        how a retriever turns a vector store's ``(chunk_id, score)`` hits
        back into ``Chunk`` objects.
        """
        found: dict[str, Chunk] = {}
        # Stay well under SQLite's bound-parameter limit for a large k.
        for start in range(0, len(chunk_ids), _IN_BATCH):
            batch = list(chunk_ids[start : start + _IN_BATCH])
            placeholders = ", ".join("?" for _ in batch)
            cursor = self._conn.execute(
                "SELECT chunk_id, doc_id, ordinal, text, start_char, end_char, "
                f"token_count, metadata FROM chunks WHERE chunk_id IN ({placeholders})",
                batch,
            )
            for row in cursor:
                chunk = _row_to_chunk(row)
                found[chunk.chunk_id] = chunk
        return found

    def filter_chunk_ids(self, filter: Filter) -> set[str]:
        """Return the ids of every chunk matching *filter*.

        The metadata **pre**-filter (plan.md §6): the result is handed to
        ``NumpyVectorStore.search(allowed_ids=...)`` so the top-*k* is taken
        over matching chunks only. A filter that matches nothing returns an
        empty set — not an error.

        Parameters
        ----------
        filter
            A mapping in the :mod:`nanorag.store.filters` grammar.

        Raises
        ------
        RetrievalError
            *filter* is not in the grammar.

        """
        where, params = compile_filter(filter)
        cursor = self._conn.execute(
            "SELECT chunks.chunk_id FROM chunks "
            "JOIN documents ON documents.doc_id = chunks.doc_id "
            f"WHERE {where}",
            params,
        )
        return {row[0] for row in cursor}

    def search_bm25(
        self,
        match_query: str,
        k: int,
        *,
        allowed_ids: Collection[str] | None = None,
    ) -> list[tuple[str, float]]:
        """Return up to *k* ``(chunk_id, score)`` pairs, highest BM25 score first.

        Parameters
        ----------
        match_query
            An FTS5 ``MATCH`` expression, already built — this method does
            no tokenizing of its own (see
            ``nanorag.retrieval.bm25._fts5_match_query``), mirroring how
            :meth:`NumpyVectorStore.search
            <nanorag.store.numpy_store.NumpyVectorStore.search>` takes an
            already-embedded query vector rather than raw text.
        k
            Maximum number of results.
        allowed_ids
            When given, restricts the search to this set of chunk ids (a
            metadata pre-filter's result), applied **before** the top-*k*
            cut, never after (plan.md §15 #7). Batched in groups of
            :data:`_IN_BATCH` to stay under SQLite's bound-parameter limit;
            merging the batches and re-sorting is exact, not approximate —
            FTS5's ``bm25()`` computes its IDF statistics over the whole
            indexed corpus, not the filtered rows in one query's result set,
            so a chunk's score does not depend on which batch it landed in.

        Raises
        ------
        StoreError
            :attr:`fts5_available` is ``False``, or *k* < 1.

        """
        if not self._fts5_available:
            raise StoreError("FTS5 is not available in this SQLite build")
        if k < 1:
            raise StoreError(f"k must be >= 1, got {k}")

        if allowed_ids is None:
            rows = self._conn.execute(
                "SELECT chunk_id, bm25(chunks_fts) FROM chunks_fts "
                "WHERE chunks_fts MATCH ? ORDER BY bm25(chunks_fts) LIMIT ?",
                (match_query, k),
            ).fetchall()
        else:
            allowed = list(allowed_ids)
            rows = []
            for start in range(0, len(allowed), _IN_BATCH):
                batch = allowed[start : start + _IN_BATCH]
                placeholders = ", ".join("?" for _ in batch)
                rows.extend(
                    self._conn.execute(
                        "SELECT chunk_id, bm25(chunks_fts) FROM chunks_fts "
                        f"WHERE chunks_fts MATCH ? AND chunk_id IN ({placeholders}) "
                        "ORDER BY bm25(chunks_fts) LIMIT ?",
                        (match_query, *batch, k),
                    ).fetchall()
                )

        # bm25() is a cost (lower is better); negate to the higher-is-better
        # convention every other score in this codebase uses, then re-sort —
        # each batch above is only individually ordered — with the same
        # score-desc/id-asc tie-break every other store uses.
        scored = sorted(
            ((chunk_id, -raw) for chunk_id, raw in rows),
            key=lambda pair: (-pair[1], pair[0]),
        )
        return scored[:k]

    # -- embeddings ----------------------------------------------------------

    def index_meta(self) -> tuple[str, int] | None:
        """Return the ``(model_id, dim)`` the index was built with, if any."""
        rows = dict(
            self._conn.execute(
                "SELECT key, value FROM meta WHERE key IN (?, ?)",
                (_MODEL_ID_KEY, _DIM_KEY),
            )
        )
        if _MODEL_ID_KEY not in rows or _DIM_KEY not in rows:
            return None
        return rows[_MODEL_ID_KEY], int(rows[_DIM_KEY])

    def upsert_embeddings(
        self, model_id: str, dim: int, chunk_ids: Sequence[str], vectors: np.ndarray
    ) -> None:
        """Persist *vectors* for *chunk_ids* under ``(model_id, dim)``.

        Parameters
        ----------
        model_id
            Identifier of the embedding model that produced *vectors*.
        dim
            Vector width. Must match every row of *vectors*.
        chunk_ids
            Chunk ids, aligned with the rows of *vectors*.
        vectors
            ``(len(chunk_ids), dim)`` array; cast to ``float32`` and stored
            as a raw bit-exact BLOB per row.

        Raises
        ------
        IndexModelMismatch
            The store already records a different ``(model_id, dim)`` — an
            index is built with one embedding model at a time.
        StoreError
            *vectors* is not shaped ``(len(chunk_ids), dim)``.

        """
        vectors = np.asarray(vectors, dtype=np.float32)
        if vectors.ndim != 2 or vectors.shape[1] != dim:
            raise StoreError(
                "vectors must have shape (n, dim)",
                dim=dim,
                shape=tuple(vectors.shape),
            )
        if len(chunk_ids) != vectors.shape[0]:
            raise StoreError(
                "chunk_ids and vectors length mismatch",
                chunk_ids=len(chunk_ids),
                vectors=vectors.shape[0],
            )

        recorded = self.index_meta()
        if recorded is not None and recorded != (model_id, dim):
            raise IndexModelMismatch(
                "embedding model/dim does not match the index",
                recorded_model_id=recorded[0],
                recorded_dim=recorded[1],
                requested_model_id=model_id,
                requested_dim=dim,
            )

        with self._conn:
            self._conn.execute(
                "INSERT INTO meta(key, value) VALUES (?, ?) "
                "ON CONFLICT(key) DO UPDATE SET value = excluded.value",
                (_MODEL_ID_KEY, model_id),
            )
            self._conn.execute(
                "INSERT INTO meta(key, value) VALUES (?, ?) "
                "ON CONFLICT(key) DO UPDATE SET value = excluded.value",
                (_DIM_KEY, str(dim)),
            )
            self._conn.executemany(
                """
                INSERT INTO embeddings(chunk_id, model_id, dim, vector)
                VALUES (?, ?, ?, ?)
                ON CONFLICT(chunk_id) DO UPDATE SET
                    model_id = excluded.model_id,
                    dim = excluded.dim,
                    vector = excluded.vector
                """,
                [
                    (cid, model_id, dim, vec.tobytes())
                    for cid, vec in zip(chunk_ids, vectors, strict=True)
                ],
            )

    def iter_embeddings(self) -> Iterator[tuple[str, np.ndarray]]:
        """Yield every stored ``(chunk_id, vector)`` pair.

        The vectors are read back bit-exact (raw ``float32`` bytes, no
        precision loss). This is how a ``NumpyVectorStore`` is rebuilt after
        a restart, without re-running the embedder.
        """
        cursor = self._conn.execute(
            "SELECT chunk_id, dim, vector FROM embeddings ORDER BY chunk_id"
        )
        for chunk_id, dim, blob in cursor:
            yield chunk_id, np.frombuffer(blob, dtype=np.float32, count=dim).copy()
