"""``FixedChunker`` — fixed-size token windows, no structural awareness.

Unlike :class:`~nanorag.chunking.recursive.RecursiveChunker`, this never looks
for paragraph, line or word boundaries — it slides a character window sized
to fit *target_tokens* straight across ``Document.text``. Simpler and faster;
prefer :class:`~nanorag.chunking.recursive.RecursiveChunker` (the plan's
default) when splitting on natural boundaries matters.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from nanorag.chunking.base import (
    default_counter,
    extend_to_token_limit,
    make_chunk,
    validate_budget,
)
from nanorag.types import Chunk, Document

if TYPE_CHECKING:
    from nanorag.tokens import TokenCounter


class FixedChunker:
    """Splits a document into fixed-size, possibly-overlapping token windows.

    Parameters
    ----------
    target_tokens
        Maximum tokens per chunk. Must be >= 1.
    overlap_tokens
        Tokens of overlap between consecutive chunks. Must be ``0 <=
        overlap_tokens < target_tokens``.
    counter
        Counts tokens for sizing. Defaults to a warning-free
        ``HeuristicCounter`` — pass a real tokenizer (e.g.
        ``TiktokenCounter``) to size chunks against an actual model.

    Raises
    ------
    ChunkingError
        The token budget is invalid (see :func:`nanorag.chunking.base.validate_budget`).

    """

    def __init__(
        self,
        *,
        target_tokens: int = 512,
        overlap_tokens: int = 64,
        counter: TokenCounter | None = None,
    ) -> None:
        """Validate the budget and store it (see the class docstring)."""
        validate_budget(target_tokens, overlap_tokens)
        self.target_tokens = target_tokens
        self.overlap_tokens = overlap_tokens
        self.counter: TokenCounter = (
            counter if counter is not None else default_counter()
        )

    def chunk(self, document: Document) -> list[Chunk]:
        """Split *document* into ordered, offset-correct fixed-size chunks."""
        text = document.text
        if not text:
            return []

        n = len(text)
        chunks: list[Chunk] = []
        ordinal = 0
        start = 0
        while start < n:
            end = extend_to_token_limit(
                text, start, n, self.target_tokens, self.counter
            )
            c = make_chunk(document, ordinal, start, end, self.counter)
            chunks.append(c)
            ordinal += 1

            if end >= n:
                start = n  # let the loop condition end the walk
            elif self.overlap_tokens > 0 and c.token_count > 0:
                overlap_chars = round(
                    (end - start) * self.overlap_tokens / c.token_count
                )
                start = max(start + 1, end - overlap_chars)
            else:
                start = end
        return chunks
