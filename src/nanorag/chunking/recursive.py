"""``RecursiveChunker`` — the default chunker (plan.md §7).

Splits on a hierarchy of separators (paragraph, then line, then sentence,
then word), recursing into finer separators only where a piece is still too
large, then greedily packs the resulting pieces into windows of up to
*target_tokens* with up to *overlap_tokens* of overlap between consecutive
windows. A piece that exhausts every separator and is still oversized (one
very long word) falls back to a character-level split.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING

from nanorag.chunking.base import (
    DEFAULT_OVERLAP_TOKENS,
    DEFAULT_SEPARATORS,
    DEFAULT_TARGET_TOKENS,
    atomic_spans,
    default_counter,
    make_chunk,
    pack_spans,
    validate_budget,
)
from nanorag.types import Chunk, Document

if TYPE_CHECKING:
    from nanorag.tokens import TokenCounter


class RecursiveChunker:
    """Splits a document along natural boundaries, packed to a token budget.

    Parameters
    ----------
    target_tokens
        Maximum tokens per chunk. Must be >= 1. The default,
        :data:`~nanorag.chunking.base.DEFAULT_TARGET_TOKENS`, was picked by
        measurement (Phase D3).
    overlap_tokens
        Tokens of overlap between consecutive chunks. Must be ``0 <=
        overlap_tokens < target_tokens``.
    separators
        Boundary hierarchy tried in order, finest fallback last. Defaults to
        ``DEFAULT_SEPARATORS`` (paragraph, line, sentence, word).
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
        target_tokens: int = DEFAULT_TARGET_TOKENS,
        overlap_tokens: int = DEFAULT_OVERLAP_TOKENS,
        separators: Sequence[str] = DEFAULT_SEPARATORS,
        counter: TokenCounter | None = None,
    ) -> None:
        """Validate the budget and store it (see the class docstring)."""
        validate_budget(target_tokens, overlap_tokens)
        self.target_tokens = target_tokens
        self.overlap_tokens = overlap_tokens
        self.separators = tuple(separators)
        self.counter: TokenCounter = (
            counter if counter is not None else default_counter()
        )

    def chunk(self, document: Document) -> list[Chunk]:
        """Split *document* into ordered, offset-correct chunks."""
        text = document.text
        if not text:
            return []

        spans = atomic_spans(
            text, 0, len(text), self.separators, self.target_tokens, self.counter
        )
        merged = pack_spans(
            text, spans, self.target_tokens, self.overlap_tokens, self.counter
        )
        return [
            make_chunk(document, ordinal, start, end, self.counter)
            for ordinal, (start, end) in enumerate(merged)
        ]
