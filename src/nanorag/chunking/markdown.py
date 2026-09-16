"""``MarkdownChunker`` — heading-aware chunking that preserves ``heading_path``.

Splits the document into sections along ATX headings (``#`` … ``######``)
first, then applies the same separator-hierarchy packing as
:class:`~nanorag.chunking.recursive.RecursiveChunker` *within* each section,
so a chunk never straddles a heading boundary. Every chunk carries the
reserved ``nanorag.heading_path`` metadata key — the ``" > "``-joined titles
of the heading and its ancestors (plan.md §11: "markdown structure preserved
as heading_path").
"""

from __future__ import annotations

import re
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

#: One ATX heading line: 1-6 ``#``s, whitespace, then the title (trailing
#: ``#``s and whitespace, if any, stripped from the captured title).
_HEADING_RE = re.compile(r"(?m)^(#{1,6})[ \t]+(\S.*?)[ \t]*#*[ \t]*$")


def _iter_sections(text: str) -> list[tuple[str, int, int]]:
    """Split *text* into ``(heading_path, start, end)`` sections.

    Each section spans from its own heading line (inclusive) up to the very
    next heading of *any* level, or the end of the text — so sections
    partition ``[0, len(text))`` exactly, excluding their descendants'
    content (any text before the first heading becomes a section with an
    empty ``heading_path``). ``heading_path`` still carries the full
    ancestry: a level-3 heading's path includes the level-1 and level-2
    headings above it even though their own *content* is a separate,
    preceding section.
    """
    matches = list(_HEADING_RE.finditer(text))
    if not matches:
        return [("", 0, len(text))]

    boundaries = [(len(m.group(1)), m.start(), m.group(2).strip()) for m in matches]

    sections: list[tuple[str, int, int]] = []
    if boundaries[0][1] > 0:
        sections.append(("", 0, boundaries[0][1]))

    stack: list[tuple[int, str]] = []
    for i, (level, start, title) in enumerate(boundaries):
        while stack and stack[-1][0] >= level:
            stack.pop()
        stack.append((level, title))

        end = boundaries[i + 1][1] if i + 1 < len(boundaries) else len(text)
        sections.append((" > ".join(t for _, t in stack), start, end))
    return sections


class MarkdownChunker:
    """Splits a markdown document by heading, then by size within each section.

    Parameters
    ----------
    target_tokens
        Maximum tokens per chunk. Must be >= 1.
    overlap_tokens
        Tokens of overlap between consecutive chunks *within the same
        section* (chunks never overlap across a heading boundary). Must be
        ``0 <= overlap_tokens < target_tokens``.
    separators
        Boundary hierarchy tried within each section. Defaults to
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
        """Split *document* into ordered, offset-correct, heading-tagged chunks."""
        text = document.text
        if not text:
            return []

        chunks: list[Chunk] = []
        ordinal = 0
        for heading_path, start, end in _iter_sections(text):
            spans = atomic_spans(
                text, start, end, self.separators, self.target_tokens, self.counter
            )
            merged = pack_spans(
                text, spans, self.target_tokens, self.overlap_tokens, self.counter
            )
            metadata = {"nanorag.heading_path": heading_path} if heading_path else {}
            for s, e in merged:
                chunks.append(
                    make_chunk(document, ordinal, s, e, self.counter, metadata)
                )
                ordinal += 1
        return chunks
