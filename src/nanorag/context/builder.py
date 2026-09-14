"""``ContextBuilder`` — fit ranked chunks into a token budget, labelled ``[1]…[n]``.

The builder's job, per plan.md §6: dedup, order, truncate, label. Returns
the blocks *and* the chunk-id → label map. Explicitly not its job: talking to
the LLM — that is ``nanorag.prompting``.

The one guarantee that matters (the C2 checkpoint, plan.md §9): the rendered
context **never exceeds the budget** under the counter it was given. It is
checked on the *joined* text the prompt will actually carry, not on a sum of
per-block counts — tokenizers merge across block boundaries, so a sum is not
the number the provider will see.

Truncation policy (plan.md §11 "Context limits"): drop lowest-ranked blocks
first. Blocks are taken in rank order until the first one that does not fit,
and nothing after it is considered — so the context is always the top-*n*
by rank for some *n*, never a top-*k* with holes in it. A chunk is never
split to make it fit: a partial chunk would no longer be the text its
``start_char``/``end_char`` describe, and citations (Phase D) point at those
offsets. Truncation is reported on ``Context.truncated``, never silent
(plan.md §15 #6).

Input order is rank order. The builder does not re-sort by score — the
retriever (and, later, the reranker or fusion stage) owns ranking and has
already broken ties deterministically; re-sorting here would undo any
deliberate reordering upstream. Presentation order is rank order too, so
``[1]`` is always the strongest hit.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from types import MappingProxyType
from typing import TYPE_CHECKING, Any

from nanorag.errors import ConfigError
from nanorag.hashing import normalize_text
from nanorag.tokens import HeuristicCounter
from nanorag.types import ScoredChunk

if TYPE_CHECKING:
    from nanorag.tokens import TokenCounter

#: Placed between consecutive rendered blocks in ``Context.text``.
BLOCK_SEPARATOR = "\n\n"


def render_block(label: int, text: str) -> str:
    """Return the prompt text for one block: its ``[label]`` line, then *text*.

    The chunk text is placed verbatim — not stripped, not escaped — so what
    ``Answer.contexts`` says was in the prompt is byte-for-byte what was in
    the prompt (plan.md §1 "Traceable").
    """
    return f"[{label}]\n{text}"


def render_context(blocks: Sequence[ContextBlock]) -> str:
    """Join rendered *blocks* with :data:`BLOCK_SEPARATOR`; ``""`` for none."""
    return BLOCK_SEPARATOR.join(block.text for block in blocks)


@dataclass(frozen=True, slots=True)
class ContextBlock:
    """One labelled chunk as it appears in the prompt.

    Attributes
    ----------
    label
        The ``[n]`` marker, 1-based and contiguous in block order.
    hit
        The scored chunk behind the block.
    text
        The rendered block — :func:`render_block` of ``label`` and the
        chunk's text.
    token_count
        Tokens in ``text`` alone, under the builder's counter. Informational:
        the budget guarantee is on the joined ``Context.text``, whose count
        can differ from the sum of these by a few boundary tokens.

    """

    label: int
    hit: ScoredChunk
    text: str
    token_count: int

    def __post_init__(self) -> None:
        """Validate the field values (see the class docstring)."""
        if self.label < 1:
            raise ValueError(f"ContextBlock.label must be >= 1, got {self.label}")
        if self.token_count < 0:
            raise ValueError(
                f"ContextBlock.token_count must be >= 0, got {self.token_count}"
            )

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serialisable dict of this block."""
        return {
            "label": self.label,
            "hit": self.hit.to_dict(),
            "text": self.text,
            "token_count": self.token_count,
        }


@dataclass(frozen=True, slots=True)
class Context:
    """What :meth:`ContextBuilder.build` produced for one query.

    Attributes
    ----------
    blocks
        The blocks that fit, in rank order (``[1]`` first).
    labels
        Read-only ``chunk_id -> label`` map — how a ``[n]`` marker in the
        answer is resolved back to a chunk (Phase D).
    text
        Every block rendered and joined — exactly what the prompt builder
        fences. ``""`` when no block fit.
    token_count
        ``counter.count(text)`` — the number the budget was checked against.
    budget_tokens
        The budget this context was built to. ``token_count <=
        budget_tokens`` always holds.
    truncated
        At least one candidate was dropped to fit the budget. ``False`` when
        every deduplicated, non-empty candidate is present — including when
        there were none.

    """

    blocks: tuple[ContextBlock, ...]
    labels: Mapping[str, int]
    text: str
    token_count: int
    budget_tokens: int
    truncated: bool

    def __post_init__(self) -> None:
        """Validate the field values (see the class docstring)."""
        if not isinstance(self.blocks, tuple):
            raise TypeError("Context.blocks must be a tuple")
        if self.token_count > self.budget_tokens:
            raise ValueError(
                f"Context.token_count ({self.token_count}) exceeds "
                f"budget_tokens ({self.budget_tokens})"
            )
        object.__setattr__(self, "labels", MappingProxyType(dict(self.labels)))

    @property
    def hits(self) -> tuple[ScoredChunk, ...]:
        """The scored chunks behind ``blocks``, in block order.

        This is what ``Answer.contexts`` records: exactly the chunks that were
        placed in the prompt (plan.md §5).
        """
        return tuple(block.hit for block in self.blocks)

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serialisable dict of this context."""
        return {
            "blocks": [b.to_dict() for b in self.blocks],
            "labels": dict(self.labels),
            "text": self.text,
            "token_count": self.token_count,
            "budget_tokens": self.budget_tokens,
            "truncated": self.truncated,
        }


class ContextBuilder:
    """Fit ranked chunks into a token budget (see the module docstring).

    Parameters
    ----------
    budget_tokens
        Maximum tokens the rendered context may occupy, under *counter*.
        Must be >= 1. The pipeline derives it from the generator's window
        minus reserved output, prompt overhead and a safety margin (plan.md
        §11) — it is never hard-coded here.
    counter
        Counts tokens for the budget check. Defaults to a warning-free
        ``HeuristicCounter``; pass the generator's real tokenizer to budget
        against an actual model (plan.md §18 F4).

    Raises
    ------
    ConfigError
        *budget_tokens* < 1.

    """

    def __init__(
        self,
        budget_tokens: int,
        *,
        counter: TokenCounter | None = None,
    ) -> None:
        """Validate the budget and store it (see the class docstring)."""
        if budget_tokens < 1:
            raise ConfigError(f"budget_tokens must be >= 1, got {budget_tokens}")
        self.budget_tokens = budget_tokens
        self.counter: TokenCounter = (
            counter if counter is not None else HeuristicCounter(warn=False)
        )

    def build(self, hits: Sequence[ScoredChunk]) -> Context:
        """Select, label and render the blocks that fit the budget.

        Parameters
        ----------
        hits
            Candidate chunks **in rank order** (best first). Order is
            preserved; nothing is re-sorted.

        Returns
        -------
        Context
            ``context.token_count <= self.budget_tokens`` always holds, and
            ``context.blocks`` is a prefix — in order — of the candidates
            that survive deduplication.

        Notes
        -----
        Deduplication keeps the first (highest-ranked) occurrence of a chunk
        id *and* of a normalised text (NFC + LF, surrounding whitespace
        ignored) — the same paragraph reached through two documents, or the
        same chunk returned twice by a fused retriever, is placed once. A
        chunk whose text is empty or whitespace-only is skipped: there is
        nothing in it to ground an answer on, and it would waste a label.
        Neither kind of skip counts as truncation.

        """
        candidates = _dedup(hits)

        blocks: list[ContextBlock] = []
        text = ""
        token_count = 0
        truncated = False
        for hit in candidates:
            label = len(blocks) + 1
            rendered = render_block(label, hit.chunk.text)
            candidate = ContextBlock(
                label=label,
                hit=hit,
                text=rendered,
                token_count=self.counter.count(rendered),
            )
            # The check is on the joined text the prompt will carry, so the
            # guarantee holds even when the counter is not additive across
            # block boundaries.
            joined = render_context([*blocks, candidate])
            joined_count = self.counter.count(joined)
            if joined_count > self.budget_tokens:
                truncated = True
                break
            blocks.append(candidate)
            text = joined
            token_count = joined_count

        return Context(
            blocks=tuple(blocks),
            labels={block.hit.chunk.chunk_id: block.label for block in blocks},
            text=text,
            token_count=token_count,
            budget_tokens=self.budget_tokens,
            truncated=truncated,
        )


def _dedup(hits: Sequence[ScoredChunk]) -> list[ScoredChunk]:
    """Drop repeated chunk ids, repeated normalised texts, and empty texts."""
    seen_ids: set[str] = set()
    seen_texts: set[str] = set()
    kept: list[ScoredChunk] = []
    for hit in hits:
        key = normalize_text(hit.chunk.text).strip()
        if not key or hit.chunk.chunk_id in seen_ids or key in seen_texts:
            continue
        seen_ids.add(hit.chunk.chunk_id)
        seen_texts.add(key)
        kept.append(hit)
    return kept
