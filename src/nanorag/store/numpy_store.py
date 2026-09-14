"""``NumpyVectorStore`` — an in-memory exact-cosine vector index.

Owns the float32 matrix and the chunk-id <-> row map (plan.md §6: "upsert,
search, delete, compact"; never text, prompts or chunking). Vectors are
assumed L2-normalised at the boundary by the embedder (plan.md §6), so
similarity is a plain dot product — exact, not approximate.

Concurrency discipline (plan.md §5, decided before Phase B): a single writer
lock serialises ``upsert``, ``delete`` and ``compact``. Every mutation builds
a new matrix / id list (and its array form) / id-map / alive-mask rather
than editing the existing ones in place, then rebinds them all together at
the end of the locked section. A ``search`` that has already captured the
previous objects (taken under the lock, in the instant before a concurrent
mutation rebinds them) therefore always finishes against one self-consistent
snapshot, never a partially-updated one — this is what makes it safe to do
the actual cosine + top-k *outside* the lock.

Ties are broken by chunk id, never by row position: row order depends on
insertion order, which differs between a live session and the same index
rebuilt from SQLite on reopen, whereas the id order is the same everywhere.
"""

from __future__ import annotations

import threading
from collections.abc import Collection, Sequence

import numpy as np

from nanorag.errors import StoreError

#: Below this fraction of live candidates, ``search`` gathers the candidate
#: rows into a copy before scoring; at or above it, it scores the whole
#: matrix in place and masks the scores (gathering copies ``n × dim`` floats,
#: which dominates the dot product itself once *n* is a sizeable share).
_GATHER_FRACTION = 0.25


class NumpyVectorStore:
    """An exact-cosine index over L2-normalised float32 vectors.

    Parameters
    ----------
    dim
        Fixed vector width for this store. Every upserted or queried vector
        must match it.

    """

    def __init__(self, dim: int) -> None:
        """Start an empty store fixed at width *dim*."""
        if dim < 1:
            raise StoreError(f"dim must be >= 1, got {dim}")
        self._dim = dim
        self._lock = threading.Lock()
        self._ids: list[str] = []
        self._id_array = np.empty((0,), dtype=str)  # ids, vectorised for search
        self._id_to_row: dict[str, int] = {}
        self._matrix = np.empty((0, dim), dtype=np.float32)
        self._alive = np.empty((0,), dtype=bool)

    @property
    def dim(self) -> int:
        """Fixed vector width of this store."""
        return self._dim

    def __len__(self) -> int:
        """Return the number of live (non-deleted) vectors."""
        return int(self._alive.sum())

    def upsert(self, chunk_ids: Sequence[str], vectors: np.ndarray) -> None:
        """Insert new vectors and overwrite existing ones, by chunk id.

        Overwriting a chunk id that was previously deleted revives it.

        Parameters
        ----------
        chunk_ids
            Chunk ids, aligned with the rows of *vectors*.
        vectors
            ``(len(chunk_ids), dim)`` array.

        Raises
        ------
        StoreError
            *vectors* is not shaped ``(len(chunk_ids), dim)``.

        """
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

        with self._lock:
            matrix = self._matrix.copy()
            alive = self._alive.copy()
            ids = list(self._ids)
            id_to_row = dict(self._id_to_row)

            new_rows: list[tuple[str, np.ndarray]] = []
            for cid, vec in zip(chunk_ids, vectors, strict=True):
                row = id_to_row.get(cid)
                if row is not None:
                    matrix[row] = vec
                    alive[row] = True
                else:
                    new_rows.append((cid, vec))

            if new_rows:
                extra = np.stack([vec for _, vec in new_rows])
                matrix = np.vstack([matrix, extra]) if matrix.shape[0] else extra
                alive = np.concatenate([alive, np.ones(len(new_rows), dtype=bool)])
                for cid, _ in new_rows:
                    id_to_row[cid] = len(ids)
                    ids.append(cid)

            self._matrix = matrix
            self._alive = alive
            self._ids = ids
            self._id_array = np.array(ids, dtype=str)
            self._id_to_row = id_to_row

    def delete(self, chunk_ids: Sequence[str]) -> None:
        """Tombstone *chunk_ids*: excluded from search, not yet reclaimed.

        Unknown chunk ids are ignored. Call :meth:`compact` to physically
        reclaim the space.
        """
        with self._lock:
            alive = self._alive.copy()
            changed = False
            for cid in chunk_ids:
                row = self._id_to_row.get(cid)
                if row is not None and alive[row]:
                    alive[row] = False
                    changed = True
            if changed:
                self._alive = alive

    def compact(self) -> None:
        """Physically remove tombstoned rows.

        The matrix, id list, id-map and alive-mask are rebound together
        (see the module docstring) so an in-flight ``search`` that already
        holds the previous snapshot is unaffected.
        """
        with self._lock:
            keep = self._alive
            if keep.all():
                return
            new_ids = [cid for cid, alive in zip(self._ids, keep, strict=True) if alive]
            new_matrix = self._matrix[keep]
            new_id_to_row = {cid: row for row, cid in enumerate(new_ids)}
            new_alive = np.ones(len(new_ids), dtype=bool)

            self._matrix = new_matrix
            self._ids = new_ids
            self._id_array = np.array(new_ids, dtype=str)
            self._id_to_row = new_id_to_row
            self._alive = new_alive

    def search(
        self,
        query: np.ndarray,
        k: int,
        *,
        allowed_ids: Collection[str] | None = None,
    ) -> list[tuple[str, float]]:
        """Return up to *k* ``(chunk_id, score)`` pairs, highest score first.

        Equal scores are ordered by chunk id, so the result is deterministic
        regardless of insertion order (see the module docstring).

        Parameters
        ----------
        query
            ``(dim,)`` vector, already L2-normalised — the score is a plain
            dot product against each stored vector.
        k
            Maximum number of results. Fewer are returned if fewer candidates
            exist.
        allowed_ids
            When given, restricts the search to this set of chunk ids (a
            metadata pre-filter's result) — expressed in id space, not
            row-index space, so a filter computed before a ``compact()`` is
            still valid after one. Deleted (tombstoned) rows are always
            excluded regardless of this filter.

        Raises
        ------
        StoreError
            *query* is not shaped ``(dim,)``, or *k* < 1.

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

        with self._lock:
            matrix = self._matrix
            ids = self._ids
            id_array = self._id_array
            alive = self._alive

        if not ids:
            return []

        mask = alive
        if allowed_ids is not None:
            allowed = set(allowed_ids)
            id_mask = np.fromiter(
                (cid in allowed for cid in ids), dtype=bool, count=len(ids)
            )
            mask = mask & id_mask

        candidate_rows = np.flatnonzero(mask)
        n = candidate_rows.size
        if n == 0:
            return []

        if n < _GATHER_FRACTION * len(ids):
            # Few candidates: copy just those rows out and score them.
            scores = matrix[candidate_rows] @ query
        else:
            # Most rows are candidates (the unfiltered case): scoring the
            # whole matrix in place is far cheaper than fancy-indexing a
            # copy of it — one dot product per row versus one copy per row.
            scores = (matrix @ query)[candidate_rows]
        if k >= n:
            contenders = np.arange(n)
        else:
            # Everything scoring at or above the k-th best is a contender —
            # including whatever ties with it at the boundary — so the
            # id-ordered sort below decides the cut, not partition order.
            kth_best = np.partition(scores, n - k)[n - k]
            contenders = np.flatnonzero(scores >= kth_best)
        # lexsort's last key is primary: score descending, then id ascending.
        order = np.lexsort((id_array[candidate_rows[contenders]], -scores[contenders]))
        return [
            (ids[candidate_rows[pos]], float(scores[pos]))
            for pos in contenders[order[:k]]
        ]
