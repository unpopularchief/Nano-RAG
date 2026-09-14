"""Persistent embedding cache, keyed by ``(model_id, sha256(normalised text))``.

Deliberately **not** keyed by ``chunk_id``: editing a document can shift every
downstream chunk's ``ordinal`` — and therefore its id — without changing that
chunk's own text, and a text-keyed cache still serves it from cache when that
happens (plan.md §5, §9 Phase B4/E2). The cache is SQLite-backed so the B4
checkpoint — "second run hits cache 100%" — holds across process restarts,
not just within one.
"""

from __future__ import annotations

import hashlib
import sqlite3
from collections.abc import Sequence
from pathlib import Path

import numpy as np

from nanorag.embeddings.base import Embedder
from nanorag.hashing import normalize_text

_SCHEMA = """
CREATE TABLE IF NOT EXISTS embedding_cache (
    model_id TEXT NOT NULL,
    text_hash TEXT NOT NULL,
    dim INTEGER NOT NULL,
    vector BLOB NOT NULL,
    PRIMARY KEY (model_id, text_hash)
);
"""


def fingerprint(text: str) -> str:
    """Return the cache key fragment for *text*: sha256 of its normalised form.

    Uses the same :func:`nanorag.hashing.normalize_text` transform as
    ``content_hash``/``chunk_id``, so a merely-re-saved (different line
    endings, canonically-equivalent unicode) chunk still hits the cache.
    """
    return hashlib.sha256(normalize_text(text).encode("utf-8")).hexdigest()


class EmbeddingCache:
    """SQLite-backed store of ``(model_id, text_hash) -> vector``.

    Parameters
    ----------
    path
        Path to the SQLite database file. Created (with its schema) if it
        does not already exist.

    """

    def __init__(self, path: str | Path) -> None:
        """Open (creating if needed) the cache database at *path*."""
        self._conn = sqlite3.connect(str(path))
        self._conn.executescript(_SCHEMA)
        self._conn.commit()

    def close(self) -> None:
        """Close the underlying SQLite connection."""
        self._conn.close()

    def __enter__(self) -> EmbeddingCache:
        """Return self, for use as a context manager."""
        return self

    def __exit__(self, *exc_info: object) -> None:
        """Close the connection on exiting the ``with`` block."""
        self.close()

    def get_many(self, model_id: str, hashes: Sequence[str]) -> dict[str, np.ndarray]:
        """Return ``{text_hash: vector}`` for whichever of *hashes* are cached."""
        if not hashes:
            return {}
        placeholders = ",".join("?" for _ in hashes)
        rows = self._conn.execute(
            f"SELECT text_hash, dim, vector FROM embedding_cache "
            f"WHERE model_id = ? AND text_hash IN ({placeholders})",
            (model_id, *hashes),
        ).fetchall()
        return {
            text_hash: np.frombuffer(blob, dtype=np.float32, count=dim).copy()
            for text_hash, dim, blob in rows
        }

    def put_many(
        self, model_id: str, dim: int, items: Sequence[tuple[str, np.ndarray]]
    ) -> None:
        """Store *items* — ``(text_hash, vector)`` pairs — under *model_id*."""
        if not items:
            return
        with self._conn:
            self._conn.executemany(
                "INSERT INTO embedding_cache(model_id, text_hash, dim, vector) "
                "VALUES (?, ?, ?, ?) "
                "ON CONFLICT(model_id, text_hash) DO UPDATE SET "
                "dim = excluded.dim, vector = excluded.vector",
                [
                    (
                        model_id,
                        text_hash,
                        dim,
                        np.asarray(vec, dtype=np.float32).tobytes(),
                    )
                    for text_hash, vec in items
                ],
            )


class CachingEmbedder:
    """Wraps an :class:`Embedder`, serving ``embed`` from a persistent cache.

    Only document embeddings are cached (keyed as described in the module
    docstring); ``embed_query`` always delegates to the wrapped embedder —
    queries are called once each, so caching buys nothing, and for some
    models a query vector is not the same as a document vector of the same
    text.

    Parameters
    ----------
    embedder
        The embedder to wrap.
    cache
        The persistent cache to read from and write to.

    """

    def __init__(self, embedder: Embedder, cache: EmbeddingCache) -> None:
        """Wrap *embedder*, checking *cache* before doing any real work."""
        self._embedder = embedder
        self._cache = cache
        self.dim = embedder.dim
        self.model_id = embedder.model_id

    def embed(self, texts: Sequence[str]) -> np.ndarray:
        """Return vectors for *texts*, computing only the cache misses."""
        texts = list(texts)
        if not texts:
            return np.empty((0, self.dim), dtype=np.float32)

        hashes = [fingerprint(t) for t in texts]
        found = self._cache.get_many(self.model_id, hashes)

        missing = [i for i, h in enumerate(hashes) if h not in found]
        if missing:
            fresh = self._embedder.embed([texts[i] for i in missing])
            new_items = [(hashes[i], fresh[j]) for j, i in enumerate(missing)]
            self._cache.put_many(self.model_id, self.dim, new_items)
            for text_hash, vector in new_items:
                found[text_hash] = np.asarray(vector, dtype=np.float32)

        return np.stack([found[h] for h in hashes]).astype(np.float32)

    def embed_query(self, text: str) -> np.ndarray:
        """Delegate to the wrapped embedder; query vectors are never cached."""
        return self._embedder.embed_query(text)
