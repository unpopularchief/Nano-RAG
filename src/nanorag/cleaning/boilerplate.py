"""Conservative, explicit boilerplate removal for extracted text.

Boilerplate cleaning must not guess away arbitrary prose. This transform
therefore removes only caller-declared whole lines plus page-edge lines that
repeat across at least three form-feed-separated pages. ``PdfLoader`` emits
that page separator; HTML callers can supply known navigation/footer lines.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Collection


def strip_boilerplate(
    text: str,
    *,
    phrases: Collection[str] = (),
    edge_lines: int = 2,
    min_repetitions: int = 3,
) -> str:
    """Remove declared lines and repeated page-edge lines from *text*.

    Matching ignores surrounding whitespace but preserves every retained
    line exactly. Page boundaries are form-feed characters and remain in the
    result, so offsets are always interpreted against the returned cleaned
    text rather than the pre-cleaned input.

    Parameters
    ----------
    text
        Extracted document text.
    phrases
        Exact whole lines to remove, after stripping surrounding whitespace.
        Useful for site-specific navigation or footer labels.
    edge_lines
        Number of nonblank lines considered at each page edge.
    min_repetitions
        Minimum pages on which an edge line must occur before removal.

    Returns
    -------
    str
        Text with only the identified boilerplate lines removed.

    Raises
    ------
    ValueError
        *edge_lines* is negative or *min_repetitions* is less than two.

    """
    if edge_lines < 0:
        raise ValueError("edge_lines must be >= 0")
    if min_repetitions < 2:
        raise ValueError("min_repetitions must be >= 2")

    pages = text.split("\f")
    repeated: set[str] = set()
    if edge_lines and len(pages) >= min_repetitions:
        counts: Counter[str] = Counter()
        for page in pages:
            nonblank = [line.strip() for line in page.splitlines() if line.strip()]
            candidates = set(nonblank[:edge_lines] + nonblank[-edge_lines:])
            counts.update(candidates)
        repeated = {line for line, count in counts.items() if count >= min_repetitions}

    removed = {phrase.strip() for phrase in phrases if phrase.strip()} | repeated
    cleaned_pages = []
    for page in pages:
        kept = [line for line in page.splitlines() if line.strip() not in removed]
        cleaned_pages.append("\n".join(kept).strip("\n"))
    return "\f".join(cleaned_pages)
