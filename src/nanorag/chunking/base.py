"""The ``Chunker`` protocol and the splitting logic shared by its implementations.

A chunker's job, precisely (plan.md §6): ``Document -> list[Chunk]`` with
correct offsets and token counts, deterministic. Not knowing about embedders
or stores.

``start_char``/``end_char`` are half-open offsets into ``Document.text``
(plan.md §5) — every helper here is written so that ``document.text[start:
end]`` always reproduces the chunk it describes, and so that the union of
every chunk's span covers the whole document with no gaps (no text lost),
regardless of how much consecutive chunks overlap.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import TYPE_CHECKING, Protocol

from nanorag.errors import ChunkingError
from nanorag.hashing import chunk_id
from nanorag.tokens import HeuristicCounter
from nanorag.types import Chunk, Document, JsonScalar

if TYPE_CHECKING:
    from nanorag.tokens import TokenCounter

#: Separator hierarchy tried in order by the structure-aware chunkers
#: (recursive, markdown): paragraph, line, sentence, word. A span that still
#: exceeds the token budget after all of these are exhausted falls back to
#: :func:`extend_to_token_limit`'s character-level bisection.
DEFAULT_SEPARATORS: tuple[str, ...] = ("\n\n", "\n", ". ", " ")

#: Default chunk budget, **picked by measurement** (Phase D3 sweep,
#: ``docs/evaluation.md``): over 128 / 256 / 512 / 1024 target tokens and
#: 0 / 32 / 64 overlap on the ``nanorag-docs`` eval set, 128-token chunks
#: with a 64-token overlap retrieved the most gold at every cut-off
#: (Recall@5 0.84 vs 0.69 for the previous 512 / 64) and gave a 7B local
#: model its best answer and citation rates. Tokens are counted by the
#: chunker's own counter — the warning-free ``HeuristicCounter`` unless one
#: is passed — so 128 is roughly 512 characters.
DEFAULT_TARGET_TOKENS = 128
DEFAULT_OVERLAP_TOKENS = 64


class Chunker(Protocol):
    """A component that splits one ``Document`` into ordered ``Chunk``s.

    Structural, not a base class (plan.md §5 rule 2) — any object with a
    matching ``chunk`` method satisfies this protocol.
    """

    def chunk(self, document: Document) -> list[Chunk]:
        """Split *document* into zero or more chunks, in document order."""
        ...


def default_counter() -> HeuristicCounter:
    """Return the chunkers' fallback counter when none is supplied.

    ``warn=False`` — a chunker constructed without an explicit
    :class:`~nanorag.tokens.TokenCounter` is a deliberate, documented default
    (plan.md §7 shows ``RecursiveChunker(target_tokens=512,
    overlap_tokens=64)`` with no counter argument), not a call site that
    should nag on every construction. Passing a real counter (e.g.
    ``TiktokenCounter``) explicitly is unaffected.
    """
    return HeuristicCounter(warn=False)


def validate_budget(target_tokens: int, overlap_tokens: int) -> None:
    """Validate a chunker's size budget.

    Raises
    ------
    ChunkingError
        *target_tokens* is not >= 1, *overlap_tokens* is negative, or
        *overlap_tokens* is not strictly less than *target_tokens* — overlap
        must never reach or exceed the size of the chunk it overlaps.

    """
    if target_tokens < 1:
        raise ChunkingError(f"target_tokens must be >= 1, got {target_tokens}")
    if overlap_tokens < 0:
        raise ChunkingError(f"overlap_tokens must be >= 0, got {overlap_tokens}")
    if overlap_tokens >= target_tokens:
        raise ChunkingError(
            f"overlap_tokens ({overlap_tokens}) must be < "
            f"target_tokens ({target_tokens})"
        )


def make_chunk(
    document: Document,
    ordinal: int,
    start: int,
    end: int,
    counter: TokenCounter,
    metadata: Mapping[str, JsonScalar] | None = None,
) -> Chunk:
    """Build the ``Chunk`` for ``document.text[start:end]``.

    Parameters
    ----------
    document
        The owning document.
    ordinal
        Zero-based position of this chunk within *document*.
    start, end
        Half-open character offsets into ``document.text``.
    counter
        Used to compute ``token_count`` for the chunk's text.
    metadata
        Extra chunk metadata (e.g. a markdown ``nanorag.heading_path``).

    """
    text = document.text[start:end]
    return Chunk(
        chunk_id=chunk_id(document.doc_id, ordinal, text),
        doc_id=document.doc_id,
        ordinal=ordinal,
        text=text,
        start_char=start,
        end_char=end,
        token_count=counter.count(text),
        metadata=metadata or {},
    )


def extend_to_token_limit(
    text: str, start: int, limit: int, target_tokens: int, counter: TokenCounter
) -> int:
    """Return the largest ``end`` in ``(start, limit]`` fitting *target_tokens*.

    Binary-searches the character offset (rather than assuming any fixed
    chars-per-token ratio) so it works for any :class:`TokenCounter`.
    Guarantees ``end > start`` — advances at least one character even when a
    single character's token count already exceeds *target_tokens* — so
    callers never loop forever on a pathologically small budget.

    Parameters
    ----------
    text
        The text being windowed.
    start
        Start offset (inclusive) of the window.
    limit
        Largest offset (exclusive bound) the window may reach.
    target_tokens
        Maximum token count the returned window may contain.
    counter
        Used to count tokens of candidate windows.

    Returns
    -------
    int
        The window's end offset, in ``(start, limit]``.

    """
    lo, hi = start + 1, limit
    best = min(start + 1, limit)
    while lo <= hi:
        mid = (lo + hi) // 2
        if counter.count(text[start:mid]) <= target_tokens:
            best = mid
            lo = mid + 1
        else:
            hi = mid - 1
    return best


def bisect_split(
    text: str, start: int, end: int, target_tokens: int, counter: TokenCounter
) -> list[tuple[int, int]]:
    """Split ``text[start:end]`` into consecutive windows of <= *target_tokens*.

    The character-level fallback used once the separator hierarchy is
    exhausted and a span still exceeds the budget (e.g. one very long word).
    """
    spans = []
    pos = start
    while pos < end:
        piece_end = extend_to_token_limit(text, pos, end, target_tokens, counter)
        spans.append((pos, piece_end))
        pos = piece_end
    return spans


def _split_span_by_sep(
    text: str, start: int, end: int, sep: str
) -> list[tuple[int, int]]:
    """Split ``text[start:end]`` on *sep*, keeping each piece contiguous.

    Each separator stays attached to the piece that precedes it, so the
    pieces partition ``[start, end)`` exactly (no gaps — nothing is lost or
    duplicated).
    """
    if not sep:
        return [(start, end)]
    spans = []
    pos = start
    while True:
        idx = text.find(sep, pos, end)
        if idx == -1:
            if pos < end:
                spans.append((pos, end))
            break
        piece_end = idx + len(sep)
        spans.append((pos, piece_end))
        pos = piece_end
    return spans


def atomic_spans(
    text: str,
    start: int,
    end: int,
    separators: tuple[str, ...],
    target_tokens: int,
    counter: TokenCounter,
) -> list[tuple[int, int]]:
    """Split ``text[start:end]`` into the finest spans usable by :func:`pack_spans`.

    Tries *separators* in order, recursing into the next separator only for
    pieces that are still too large — the standard "recursive" splitting
    strategy (paragraph, then line, then sentence, then word). A span that
    exhausts every separator and is still oversized (e.g. one long word)
    falls back to :func:`bisect_split`. Every returned span independently
    satisfies ``counter.count(text[s:e]) <= target_tokens`` and the spans
    partition ``[start, end)`` exactly.
    """
    if end <= start:
        return []
    if counter.count(text[start:end]) <= target_tokens:
        return [(start, end)]
    if not separators:
        return bisect_split(text, start, end, target_tokens, counter)

    sep, *rest = separators
    pieces = _split_span_by_sep(text, start, end, sep)
    if len(pieces) <= 1:
        # This separator did not actually split anything; try the next one.
        return atomic_spans(text, start, end, tuple(rest), target_tokens, counter)

    result: list[tuple[int, int]] = []
    for piece_start, piece_end in pieces:
        result.extend(
            atomic_spans(
                text, piece_start, piece_end, tuple(rest), target_tokens, counter
            )
        )
    return result


def pack_spans(
    text: str,
    spans: list[tuple[int, int]],
    target_tokens: int,
    overlap_tokens: int,
    counter: TokenCounter,
) -> list[tuple[int, int]]:
    """Greedily merge consecutive *spans* into windows of <= *target_tokens*.

    *spans* must already partition a contiguous range with no gaps (as
    :func:`atomic_spans` returns). Consecutive output windows overlap by up
    to *overlap_tokens*, chosen by resuming from the first input span whose
    start reaches that far back into the window just emitted — so the next
    window always starts strictly after the previous window's start,
    guaranteeing termination and never regressing the overall position.
    """
    if not spans:
        return []
    chunks: list[tuple[int, int]] = []
    n = len(spans)
    i = 0
    while i < n:
        start = spans[i][0]
        end = spans[i][1]
        j = i
        while (
            j + 1 < n and counter.count(text[start : spans[j + 1][1]]) <= target_tokens
        ):
            j += 1
            end = spans[j][1]
        chunks.append((start, end))
        if j + 1 >= n:
            i = n  # let the loop condition end the walk
            continue

        chunk_tokens = counter.count(text[start:end])
        if overlap_tokens > 0 and chunk_tokens > 0:
            overlap_chars = round((end - start) * overlap_tokens / chunk_tokens)
            target_next_start = end - overlap_chars
        else:
            target_next_start = end

        next_i = i + 1
        while next_i <= j and spans[next_i][0] < target_next_start:
            next_i += 1
        i = next_i
    return chunks
