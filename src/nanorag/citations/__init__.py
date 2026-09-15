"""Citation resolution: a generated answer's ``[n]`` markers -> ``Citation``s.

One module so far, ``parser.py``; ``verify.py`` from the eventual layout
(plan.md §8) splits out only when a second citation-time concern (beyond
resolving markers) exists. Pure Python over the core types — no ``numpy``,
no provider.
"""

from nanorag.citations.parser import MARKER_PATTERN, CitationReport, parse_citations

__all__ = [
    "MARKER_PATTERN",
    "CitationReport",
    "parse_citations",
]
