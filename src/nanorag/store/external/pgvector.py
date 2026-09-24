"""``PgVectorStore`` — a ``VectorStore`` backed by a Postgres/pgvector table.

``pg8000`` is imported lazily, inside ``__init__`` only, so importing
``nanorag.store.external`` never requires the ``[pgvector]`` extra — only
constructing a ``PgVectorStore`` does, mirroring
``embeddings.local.FastEmbedEmbedder`` (plan.md §3).

Driver choice: ``pg8000``, not ``psycopg``. ``psycopg`` (v3) is LGPL and
needs the system ``libpq``; ``pg8000`` is pure Python and BSD-3-Clause, the
same reasoning already applied to every other optional dependency in this
project (``pypdf``, ``selectolax``). Vectors are sent and read back as a
``'[0.1,0.2,...]'`` text literal cast to ``vector`` in SQL — this avoids
needing the separate ``pgvector`` Python package, whose numpy adapter only
registers with ``psycopg``/``asyncpg``, not ``pg8000``.

No ANN index (``CREATE INDEX ... USING hnsw``) is created on the table.
Without one, pgvector does an exact sequential-scan KNN — matching
``NumpyVectorStore``'s exact search, which is what the store conformance
suite needs (plan.md §18 F7: behaviour parity, not recall parity). A caller
who wants ANN speed can add their own index directly in Postgres.
"""

from __future__ import annotations

import re
from collections.abc import Collection, Sequence
from typing import TYPE_CHECKING, Any
from urllib.parse import unquote, urlsplit

import numpy as np

from nanorag.errors import ConfigError, StoreError

if TYPE_CHECKING:
    from pg8000.dbapi import Connection

#: Chunk ids per ``IN (...)`` query, mirroring
#: ``store.sqlite_docs._IN_BATCH``'s reasoning (stay well under the
#: driver/server's bound-parameter limit for a large candidate set).
_IN_BATCH = 500

_IDENTIFIER_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")


def _parse_dsn(dsn: str) -> dict[str, Any]:
    """Turn a ``postgresql://user:pass@host:port/db`` DSN into connect kwargs.

    ``pg8000.dbapi.connect`` takes keyword arguments, not a DSN string.
    """
    parsed = urlsplit(dsn)
    if parsed.scheme not in ("postgres", "postgresql"):
        raise StoreError(
            "dsn must be a postgresql:// URL", scheme=parsed.scheme or None
        )
    if not parsed.username:
        raise StoreError("dsn must include a user")
    return {
        "user": unquote(parsed.username),
        "password": unquote(parsed.password) if parsed.password else None,
        "host": parsed.hostname or "localhost",
        "port": parsed.port or 5432,
        "database": parsed.path.lstrip("/") or None,
    }


def _vector_literal(vector: np.ndarray) -> str:
    """Return *vector* as pgvector's ``'[x,y,...]'`` text literal."""
    return "[" + ",".join(repr(float(x)) for x in vector) + "]"


def _parse_vector(text: str) -> np.ndarray:
    """Parse pgvector's ``'[x,y,...]'`` text literal back to a float32 array."""
    return np.array(text.strip("[]").split(","), dtype=np.float32)


class PgVectorStore:
    """A ``VectorStore`` over a Postgres table with a pgvector column.

    Parameters
    ----------
    dim
        Fixed vector width. The table is created with this width if it does
        not already exist; an existing table with a different width raises
        ``StoreError``.
    dsn
        A ``postgresql://user:pass@host:port/db`` connection string.
    table_name
        The table to use. Must be a bare SQL identifier (validated — it
        cannot be passed as a bound parameter).

    Raises
    ------
    ConfigError
        The ``[pgvector]`` extra is not installed.
    StoreError
        *table_name* is not a bare identifier, the server is unreachable,
        or an existing table's vector width does not match *dim*.

    """

    def __init__(
        self, dim: int, *, dsn: str, table_name: str = "nanorag_vectors"
    ) -> None:
        """Connect via *dsn* and ensure *table_name* exists at width *dim*."""
        if dim < 1:
            raise StoreError(f"dim must be >= 1, got {dim}")
        if not _IDENTIFIER_RE.match(table_name):
            raise StoreError(
                "table_name must be a bare SQL identifier", table_name=table_name
            )

        try:
            import pg8000.dbapi as dbapi
        except ImportError as exc:
            raise ConfigError(
                "the pgvector store needs the [pgvector] extra: run "
                '`uv add "nanorag[pgvector]"` (installs pg8000)'
            ) from exc

        try:
            conn: Connection = dbapi.connect(**_parse_dsn(dsn))
            cursor = conn.cursor()
            cursor.execute("CREATE EXTENSION IF NOT EXISTS vector")
            cursor.execute(
                f"CREATE TABLE IF NOT EXISTS {table_name} "
                f"(chunk_id TEXT PRIMARY KEY, embedding VECTOR({dim}))"
            )
            cursor.execute(
                "SELECT atttypmod FROM pg_attribute "
                "WHERE attrelid = %s::regclass AND attname = 'embedding'",
                (table_name,),
            )
            row = cursor.fetchone()
            conn.commit()
        except Exception as exc:
            raise StoreError(f"could not open pgvector store: {exc}", dsn=dsn) from exc
        if row is not None and row[0] not in (-1, dim):
            raise StoreError(
                "existing pgvector table has a different vector width",
                table_name=table_name,
                existing_dim=row[0],
                requested_dim=dim,
            )

        self._dim = dim
        self._table = table_name
        self._conn = conn

    @property
    def dim(self) -> int:
        """Fixed vector width of this store."""
        return self._dim

    def upsert(self, chunk_ids: Sequence[str], vectors: np.ndarray) -> None:
        """Insert new vectors and overwrite existing ones, by chunk id."""
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

        cursor = self._conn.cursor()
        cursor.executemany(
            f"INSERT INTO {self._table} (chunk_id, embedding) "
            "VALUES (%s, %s::vector) ON CONFLICT (chunk_id) DO UPDATE SET "
            "embedding = EXCLUDED.embedding",
            [
                (cid, _vector_literal(vec))
                for cid, vec in zip(chunk_ids, vectors, strict=True)
            ],
        )
        self._conn.commit()

    def delete(self, chunk_ids: Sequence[str]) -> None:
        """Remove *chunk_ids* from search. Unknown chunk ids are ignored."""
        if not chunk_ids:
            return
        cursor = self._conn.cursor()
        for start in range(0, len(chunk_ids), _IN_BATCH):
            batch = chunk_ids[start : start + _IN_BATCH]
            placeholders = ", ".join("%s" for _ in batch)
            cursor.execute(
                f"DELETE FROM {self._table} WHERE chunk_id IN ({placeholders})",
                list(batch),
            )
        self._conn.commit()

    def get_vectors(self, chunk_ids: Sequence[str]) -> dict[str, np.ndarray]:
        """Return the stored (live) vectors for *chunk_ids*, keyed by id."""
        if not chunk_ids:
            return {}
        cursor = self._conn.cursor()
        found: dict[str, np.ndarray] = {}
        for start in range(0, len(chunk_ids), _IN_BATCH):
            batch = chunk_ids[start : start + _IN_BATCH]
            placeholders = ", ".join("%s" for _ in batch)
            cursor.execute(
                f"SELECT chunk_id, embedding::text FROM {self._table} "
                f"WHERE chunk_id IN ({placeholders})",
                list(batch),
            )
            for cid, embedding_text in cursor.fetchall():
                found[cid] = _parse_vector(embedding_text)
        return found

    def search(
        self,
        query: np.ndarray,
        k: int,
        *,
        allowed_ids: Collection[str] | None = None,
    ) -> list[tuple[str, float]]:
        """Return up to *k* ``(chunk_id, score)`` pairs, highest score first.

        Batched the same way as :meth:`get_vectors` when *allowed_ids* is
        given; each batch's own top-*k* is merged and re-sorted (score
        descending, ties broken by chunk id) to the true global top-*k* —
        exact, not approximate, the same guarantee ``NumpyVectorStore``
        makes and ``sqlite_docs.SqliteDocumentStore.search_bm25`` already
        establishes for a batched pre-filter.
        """
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

        literal = _vector_literal(query)
        cursor = self._conn.cursor()
        batches: list[Sequence[str] | None]
        if allowed_ids is None:
            batches = [None]
        else:
            id_list = list(allowed_ids)
            batches = [
                id_list[start : start + _IN_BATCH]
                for start in range(0, len(id_list), _IN_BATCH)
            ]

        rows: list[tuple[str, float]] = []
        for batch in batches:
            where = ""
            params: list[Any] = [literal]
            if batch is not None:
                placeholders = ", ".join("%s" for _ in batch)
                where = f"WHERE chunk_id IN ({placeholders})"
                params.extend(batch)
            params.extend([literal, k])
            cursor.execute(
                f"SELECT chunk_id, -(embedding <#> %s::vector) AS score "
                f"FROM {self._table} {where} "
                "ORDER BY embedding <#> %s::vector LIMIT %s",
                params,
            )
            rows.extend((cid, float(score)) for cid, score in cursor.fetchall())

        rows.sort(key=lambda row: (-row[1], row[0]))
        return rows[:k]

    def compact(self) -> None:
        """No-op: Postgres autovacuum reclaims deleted-row storage on its own."""
