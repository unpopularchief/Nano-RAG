"""Gitignore-style glob matching against POSIX-relative paths.

Deliberately not ``fnmatch``: ``fnmatch``'s ``*`` matches across ``/``, which
makes ``"**/*.md"`` require a literal slash and therefore fail to match a
top-level ``top.md`` — surprising given the plan's own quickstart example
(``rag.ingest_path("docs/", glob="**/*.md")``, plan.md §7). Here, ``*`` stays
within one path segment and ``**`` spans segments, matching the glob
semantics most tools (git, pathlib post-3.13) already use.
"""

from __future__ import annotations

import re
from functools import lru_cache

_TOKEN = re.compile(r"\*\*/|\*\*|\*|\?|[^*?]+")


@lru_cache(maxsize=256)
def _compile(pattern: str) -> re.Pattern[str]:
    """Translate a glob *pattern* to a compiled, fully-anchored regex."""
    parts = []
    for token in _TOKEN.findall(pattern):
        if token == "**/":
            parts.append("(?:.*/)?")
        elif token == "**":
            parts.append(".*")
        elif token == "*":
            parts.append("[^/]*")
        elif token == "?":
            parts.append("[^/]")
        else:
            parts.append(re.escape(token))
    return re.compile("(?s:" + "".join(parts) + ")")


def glob_match(path: str, pattern: str) -> bool:
    """Return True if POSIX-relative *path* matches glob *pattern*.

    ``*`` matches within one path segment; ``**`` (optionally followed by
    ``/``) matches zero or more segments, so ``"**/*.md"`` matches both
    ``"a.md"`` and ``"sub/a.md"``; ``"*.md"`` matches only ``"a.md"``.
    """
    return _compile(pattern).fullmatch(path) is not None
