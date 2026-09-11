"""Splits a ``Document`` into ``Chunk``s.

Reached via ``nanorag.chunking``, not the top level, matching
``nanorag.loaders`` (plan.md §7 "Import cost").
"""

from nanorag.chunking.base import DEFAULT_SEPARATORS, Chunker
from nanorag.chunking.fixed import FixedChunker
from nanorag.chunking.markdown import MarkdownChunker
from nanorag.chunking.recursive import RecursiveChunker

__all__ = [
    "Chunker",
    "DEFAULT_SEPARATORS",
    "FixedChunker",
    "RecursiveChunker",
    "MarkdownChunker",
]
