"""Persistent storage: durable documents/chunks/vectors and the in-memory index.

Reached via ``nanorag.store``, not the top level, so ``import nanorag`` stays
free of ``numpy`` (plan.md §7 "Import cost" — the same rule applied to
``nanorag.loaders`` and ``nanorag.chunking``).
"""

from nanorag.store.numpy_store import NumpyVectorStore
from nanorag.store.sqlite_docs import SqliteDocumentStore

__all__ = ["SqliteDocumentStore", "NumpyVectorStore"]
