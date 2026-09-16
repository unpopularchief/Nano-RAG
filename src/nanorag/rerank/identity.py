"""``IdentityReranker`` — the default: no re-scoring, just a slice.

plan.md §6: "Identity reranker is the default." Not a placeholder — the
deliberate default until a real reranker measurably beats it on the Phase D
eval set (plan.md §9 Phase F Acceptance: "enabled by default only if it
improves the Phase D metrics"). With this reranker, ``retrieve_k`` and
``k`` doing the same thing is a no-op by construction: a caller who never
touches reranking at all sees byte-identical behaviour to before Phase F.
"""

from __future__ import annotations

from nanorag.types import ScoredChunk


class IdentityReranker:
    """Returns the first ``top_n`` hits unchanged — no re-scoring, ever fails."""

    def rerank(
        self, query: str, hits: list[ScoredChunk], top_n: int
    ) -> list[ScoredChunk]:
        """Return ``hits[:top_n]``; *query* is unused. Source/score untouched."""
        return hits[:top_n]
