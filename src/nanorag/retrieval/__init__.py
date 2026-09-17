"""Retrieval: query -> ranked ``ScoredChunk``s, with metadata pre-filtering.

Three implementations of the ``Retriever`` protocol (plan.md §15 #1, written
now that a second and third exist): ``DenseRetriever`` (exact cosine, Phase
C), ``Bm25Retriever`` (lexical, SQLite FTS5 with a NumPy fallback) and
``HybridRetriever`` (RRF fusion of any two or more) — the latter two from
Phase F session F2. Reached via ``nanorag.retrieval``, not the top level, so
``import nanorag`` stays free of ``numpy``.
"""

from nanorag.retrieval.base import Retriever
from nanorag.retrieval.bm25 import Bm25Retriever
from nanorag.retrieval.dense import DenseRetriever
from nanorag.retrieval.hybrid import HybridRetriever

__all__ = ["Bm25Retriever", "DenseRetriever", "HybridRetriever", "Retriever"]
