"""Optional external ``VectorStore`` backends (Phase G, session G2).

Reached via ``nanorag.store.external``, never the top level or even
``nanorag.store`` itself, so importing either of those never requires
``qdrant-client`` or ``pg8000`` (plan.md §7 "Import cost"). Each adapter also
defers its own third-party import to inside ``__init__``, so even importing
this subpackage costs nothing until a store is actually constructed.
"""

from nanorag.store.external.pgvector import PgVectorStore
from nanorag.store.external.qdrant import QdrantVectorStore

__all__ = ["QdrantVectorStore", "PgVectorStore"]
