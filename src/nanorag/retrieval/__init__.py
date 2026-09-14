"""Retrieval: query -> ranked ``ScoredChunk``s, with metadata pre-filtering.

One implementation so far, ``DenseRetriever`` (exact cosine over the NumPy
index). A ``Retriever`` protocol arrives with the second implementation
(BM25, Phase F), per the "a Protocol is written at the second implementation"
rule (plan.md §15 #1). Reached via ``nanorag.retrieval``, not the top level,
so ``import nanorag`` stays free of ``numpy``.
"""

from nanorag.retrieval.dense import DenseRetriever

__all__ = ["DenseRetriever"]
