"""Retrieval: query -> ranked ``ScoredChunk``s, with metadata pre-filtering.

``DenseRetriever`` (exact cosine, Phase C), ``Bm25Retriever`` (lexical,
SQLite FTS5 with a NumPy fallback) and ``HybridRetriever`` (RRF fusion of
any two or more) are the three implementations of the ``Retriever``
protocol from Phase F session F2. Session F3 adds three composable
wrappers over any ``Retriever``: ``MmrRetriever`` (diversify by Maximal
Marginal Relevance), ``ParentExpandingRetriever`` (widen each hit to its
neighbouring chunks) and ``MultiQueryRetriever`` (retrieve for several
phrasings of one question, fused by RRF — see ``QueryTransform``,
``IdentityQueryTransform``, ``LLMQueryTransform``). Reached via
``nanorag.retrieval``, not the top level, so ``import nanorag`` stays free
of ``numpy``.
"""

from nanorag.retrieval.base import Retriever
from nanorag.retrieval.bm25 import Bm25Retriever
from nanorag.retrieval.dense import DenseRetriever
from nanorag.retrieval.hybrid import HybridRetriever
from nanorag.retrieval.mmr import MmrRetriever
from nanorag.retrieval.parent import ParentExpandingRetriever
from nanorag.retrieval.query_transform import (
    IdentityQueryTransform,
    LLMQueryTransform,
    MultiQueryRetriever,
    QueryTransform,
)

__all__ = [
    "Bm25Retriever",
    "DenseRetriever",
    "HybridRetriever",
    "IdentityQueryTransform",
    "LLMQueryTransform",
    "MmrRetriever",
    "MultiQueryRetriever",
    "ParentExpandingRetriever",
    "QueryTransform",
    "Retriever",
]
