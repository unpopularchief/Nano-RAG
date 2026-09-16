"""Rank metrics over one query's retrieved list — pure functions, no I/O.

The unit of relevance is a **gold span** (see ``datasets.py``): a retrieved
chunk is relevant when it overlaps at least one of the item's gold spans,
and a gold span is *covered* when at least one retrieved chunk overlaps it.
Because gold is defined on the document's character offsets and not on
chunk ids, the same dataset grades every chunker, chunk size and overlap
alike — which is what makes the Phase D sweeps comparable (plan.md §9 D3).

Every function takes the ranked list as ``ranked``: for each retrieved
chunk, in rank order, the set of gold-span indices it covers (empty when
it is irrelevant). ``n_gold`` is the item's number of gold spans. A query
with **no** gold spans has no defined recall, precision, MRR or nDCG —
these raise ``ValueError`` rather than return a misleading ``0.0``; the
runner keeps unanswerable items in their own population and measures
abstention on them instead.

Definitions (``k`` counts positions, and positions beyond the list are
irrelevant):

- ``recall@k``    — covered gold spans within the top ``k`` / ``n_gold``.
- ``precision@k`` — relevant positions within the top ``k`` / ``k``.
- ``hit@k``       — ``1.0`` if any of the top ``k`` is relevant, else ``0.0``.
- ``reciprocal_rank`` — ``1 / rank`` of the first relevant position, ``0.0``
  if none.
- ``ndcg@k``      — ``DCG@k / IDCG@k`` with a gain of ``1`` at every
  position that covers a gold span **no earlier position covered** (a
  second chunk over the same span adds nothing — with overlapping chunks
  two positions often cover one span, and counting both would push nDCG
  past ``1``); the ideal list holds ``min(n_gold, k)`` such positions.
"""

from __future__ import annotations

import math
from collections.abc import Sequence

#: Which gold spans one retrieved chunk overlaps; empty means irrelevant.
Coverage = frozenset[int]


def _check(ranked: Sequence[Coverage], n_gold: int | None, k: int | None) -> None:
    if n_gold is not None and n_gold < 1:
        raise ValueError("metric is undefined for a query with no gold spans")
    if k is not None and k < 1:
        raise ValueError(f"k must be >= 1, got {k}")
    for position, covered in enumerate(ranked):
        if n_gold is not None and any(i < 0 or i >= n_gold for i in covered):
            raise ValueError(f"rank {position + 1} covers a gold index out of range")


def recall_at_k(ranked: Sequence[Coverage], n_gold: int, k: int) -> float:
    """Fraction of the *n_gold* spans covered by the top *k* results."""
    _check(ranked, n_gold, k)
    covered: set[int] = set()
    for coverage in ranked[:k]:
        covered.update(coverage)
    return len(covered) / n_gold


def precision_at_k(ranked: Sequence[Coverage], k: int) -> float:
    """Fraction of the top *k* positions that are relevant (missing = irrelevant)."""
    _check(ranked, None, k)
    return sum(1 for coverage in ranked[:k] if coverage) / k


def hit_at_k(ranked: Sequence[Coverage], k: int) -> float:
    """``1.0`` if any of the top *k* results is relevant, else ``0.0``."""
    _check(ranked, None, k)
    return 1.0 if any(ranked[:k]) else 0.0


def reciprocal_rank(ranked: Sequence[Coverage]) -> float:
    """``1 / rank`` of the first relevant result; ``0.0`` when none is."""
    _check(ranked, None, None)
    for position, coverage in enumerate(ranked, start=1):
        if coverage:
            return 1.0 / position
    return 0.0


def ndcg_at_k(ranked: Sequence[Coverage], n_gold: int, k: int) -> float:
    """Return nDCG over the top *k*: gain 1 per position covering a new gold span."""
    _check(ranked, n_gold, k)
    seen: set[int] = set()
    dcg = 0.0
    for position, coverage in enumerate(ranked[:k], start=1):
        if coverage - seen:
            dcg += 1.0 / math.log2(position + 1)
            seen.update(coverage)
    ideal = sum(
        1.0 / math.log2(position + 1) for position in range(1, min(n_gold, k) + 1)
    )
    return dcg / ideal


def coverage_of(
    doc_id: str,
    start_char: int,
    end_char: int,
    gold_spans: Sequence[tuple[str, int, int]],
) -> Coverage:
    """Return the indices of the *gold_spans* a chunk overlaps.

    *gold_spans* are ``(doc_id, start, end)`` triples; the chunk is
    ``[start_char, end_char)`` of document *doc_id*.
    """
    return frozenset(
        i
        for i, (gold_doc, gold_start, gold_end) in enumerate(gold_spans)
        if gold_doc == doc_id and start_char < gold_end and gold_start < end_char
    )
