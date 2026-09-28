"""Optional NumPy index snapshot, validated against SQLite before use."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

import numpy as np

from nanorag.store.numpy_store import NumpyVectorStore
from nanorag.store.sqlite_docs import SqliteDocumentStore


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def save_snapshot(docs: SqliteDocumentStore, path: str | Path) -> None:
    """Write a matrix and manifest; SQLite remains the durable source of truth."""
    target = Path(path)
    revision = docs.embedding_revision()
    rows = list(docs.iter_embeddings())
    meta = docs.index_meta()
    dim = meta[1] if meta is not None else 0
    matrix = (
        np.stack([vector for _, vector in rows])
        if rows
        else np.empty((0, dim), dtype=np.float32)
    )
    matrix_tmp = target.with_name(target.name + ".tmp")
    manifest = target.with_name(target.name + ".json")
    manifest_tmp = manifest.with_name(manifest.name + ".tmp")
    with matrix_tmp.open("wb") as stream:
        np.save(stream, matrix, allow_pickle=False)
    if revision != docs.embedding_revision():
        matrix_tmp.unlink()
        raise RuntimeError("embeddings changed while saving the snapshot")
    digest = _sha256(matrix_tmp)
    manifest_tmp.write_text(
        json.dumps(
            {
                "model": meta,
                "revision": revision,
                "ids": [chunk_id for chunk_id, _ in rows],
                "sha256": digest,
            }
        ),
        encoding="utf-8",
    )
    os.replace(matrix_tmp, target)
    os.replace(manifest_tmp, manifest)


def load_snapshot(
    docs: SqliteDocumentStore, dim: int, path: str | Path
) -> NumpyVectorStore | None:
    """Return a current snapshot, or ``None`` so the caller rebuilds from SQLite."""
    target = Path(path)
    manifest = target.with_name(target.name + ".json")
    try:
        info = json.loads(manifest.read_text(encoding="utf-8"))
        if (
            info["model"] != list(docs.index_meta() or ())
            or info["revision"] != docs.embedding_revision()
        ):
            return None
        if _sha256(target) != info["sha256"]:
            return None
        matrix = np.load(target, allow_pickle=False, mmap_mode="r")
        ids = info["ids"]
        if (
            not isinstance(ids, list)
            or not all(isinstance(chunk_id, str) for chunk_id in ids)
            or len(set(ids)) != len(ids)
            or matrix.dtype != np.float32
            or matrix.shape != (len(ids), dim)
        ):
            return None
        store = NumpyVectorStore(dim)
        store.upsert(ids, matrix)
        return store
    except (OSError, ValueError, KeyError, TypeError, EOFError):
        return None
