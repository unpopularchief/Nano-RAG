"""``parse_citations`` — resolve a generated answer's ``[n]`` markers to ``Citation``s.

The parser's job, per plan.md §6: extract ``[n]``, map to chunk ids (via the
``Context`` the answer was generated from), drop unresolvable markers, and
report a validity rate. Explicitly not its job: judging factual correctness
— a citation proves provenance, not truth (plan.md §11).

Markers follow the system prompt's documented convention (plan.md
``prompting/templates.py``): one label per bracket, e.g. ``[1]`` or
``[2][3]``, never ``[1,2]`` or ``[1-3]`` — those are not markers under this
parser and are neither counted nor resolved. A marker is *resolvable* when
its number is a real block label in *context*; anything else (out of range,
zero, or simply never in this context) is dropped from
``CitationReport.citations`` but still counted in ``total_markers`` for the
validity rate. ``Answer.citations`` is populated with the result, sorted by
label — the same order the ``Citation`` docstring promises — even though a
model may cite ``[3]`` before ``[1]`` in its text; duplicate markers for a
label already resolved collapse to the one ``Citation``.

Metadata propagation (the D1 checkpoint's other half): a resolved
``Citation``'s ``doc_id``/``source_uri``/``start_char``/``end_char`` trace a
straight line back through ``ScoredChunk.chunk`` (``doc_id``, the
document-relative ``start_char``/``end_char``) to the ``Document`` fetched by
*get_document* (``source_uri``) — the same chain the chunkers and stores
built in Phase B, finally consumed here.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from nanorag.errors import StoreError
from nanorag.types import Citation

if TYPE_CHECKING:
    from collections.abc import Callable

    from nanorag.context.builder import Context
    from nanorag.types import Document

#: One label per bracket — ``[12]`` matches, ``[1,2]`` and ``[1-3]`` do not.
MARKER_PATTERN = re.compile(r"\[(\d+)\]")


@dataclass(frozen=True, slots=True)
class CitationReport:
    """What :func:`parse_citations` found in one piece of generated text.

    Attributes
    ----------
    citations
        Resolved, deduplicated citations, sorted by ``label``.
    total_markers
        Every ``[n]``-shaped marker found in the text, including duplicates
        and unresolvable ones.
    resolved_markers
        How many of those occurrences resolved to a real block label
        (repeats of an already-resolved label count too — a model that
        cites the same true source five times has not made five mistakes).

    """

    citations: tuple[Citation, ...]
    total_markers: int
    resolved_markers: int

    def __post_init__(self) -> None:
        """Validate the field values (see the class docstring)."""
        if not isinstance(self.citations, tuple):
            raise TypeError("CitationReport.citations must be a tuple")
        if self.total_markers < 0:
            raise ValueError("CitationReport.total_markers must be >= 0")
        if self.resolved_markers < 0 or self.resolved_markers > self.total_markers:
            raise ValueError(
                "CitationReport.resolved_markers must be in "
                f"[0, {self.total_markers}], got {self.resolved_markers}"
            )

    @property
    def validity_rate(self) -> float:
        """Fraction of markers that resolved; ``1.0`` when there were none."""
        if self.total_markers == 0:
            return 1.0
        return self.resolved_markers / self.total_markers

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serialisable dict of this report."""
        return {
            "citations": [c.to_dict() for c in self.citations],
            "total_markers": self.total_markers,
            "resolved_markers": self.resolved_markers,
            "validity_rate": self.validity_rate,
        }


def parse_citations(
    text: str,
    context: Context,
    *,
    get_document: Callable[[str], Document | None],
) -> CitationReport:
    """Extract and resolve every ``[n]`` marker in *text* against *context*.

    Parameters
    ----------
    text
        Generated answer text to scan for markers.
    context
        The ``Context`` the answer was generated from — its ``blocks`` give
        the only labels that can resolve.
    get_document
        Resolves a chunk's ``doc_id`` to its ``Document`` (a document
        store's ``get_document``, unwrapped — no store type is imported
        here).

    Returns
    -------
    CitationReport

    Raises
    ------
    StoreError
        A block's ``doc_id`` is not in the document store. The per-document
        SQLite transaction commits before a chunk is ever placed in a
        prompt, so this should be impossible; raised loudly rather than
        dropped silently, matching ``DenseRetriever``'s phantom-chunk-id
        guard (plan.md §9 C1).

    """
    by_label = {block.label: block.hit for block in context.blocks}

    total = 0
    resolved_markers = 0
    citations: dict[int, Citation] = {}
    for match in MARKER_PATTERN.finditer(text):
        total += 1
        label = int(match.group(1))
        hit = by_label.get(label)
        if hit is None:
            continue
        resolved_markers += 1
        if label in citations:
            continue
        document = get_document(hit.chunk.doc_id)
        if document is None:
            raise StoreError(
                "chunk's document is not in the document store",
                doc_id=hit.chunk.doc_id,
                chunk_id=hit.chunk.chunk_id,
            )
        citations[label] = Citation(
            label=label,
            chunk_id=hit.chunk.chunk_id,
            doc_id=hit.chunk.doc_id,
            source_uri=document.source_uri,
            start_char=hit.chunk.start_char,
            end_char=hit.chunk.end_char,
        )

    return CitationReport(
        citations=tuple(citations[label] for label in sorted(citations)),
        total_markers=total,
        resolved_markers=resolved_markers,
    )
