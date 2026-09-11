"""Token counting behind a one-method ``Protocol``.

Context budgeting needs a token count, and the right counter depends on the
generator: ``tiktoken`` is exact only for OpenAI-family tokenizers, which are
*not* the default generators (plan.md §3, §18 F4). So counting is a swappable
strategy — :class:`TokenCounter` — and a generator picks the counter that
matches its tokenizer.

``TiktokenCounter`` imports ``tiktoken`` lazily on first use and reads its
encoding from ``TIKTOKEN_CACHE_DIR`` (or an explicit ``cache_dir``), so
``import nanorag`` stays free of it and tests never reach the network.
"""

from __future__ import annotations

import os
import warnings
from pathlib import Path
from typing import TYPE_CHECKING, Protocol, runtime_checkable

if TYPE_CHECKING:
    import tiktoken


@runtime_checkable
class TokenCounter(Protocol):
    """Something that counts the tokens in a string.

    The single extension point for tokenisation. Any object with a matching
    ``count`` method satisfies it — no base class to inherit.
    """

    def count(self, text: str) -> int:
        """Return the number of tokens *text* encodes to."""
        ...


class TiktokenCounter:
    """Exact token count for an OpenAI-family (``tiktoken``) encoding.

    Parameters
    ----------
    encoding
        A ``tiktoken`` encoding name (``"cl100k_base"``, ``"o200k_base"``).
    cache_dir
        Directory holding the vendored encoding files. When given it is set as
        ``TIKTOKEN_CACHE_DIR`` before the encoder loads, so no download
        happens. When omitted, ``tiktoken``'s own resolution applies (which
        already honours a ``TIKTOKEN_CACHE_DIR`` set by the caller or the test
        harness).

    """

    def __init__(
        self,
        encoding: str = "cl100k_base",
        *,
        cache_dir: str | Path | None = None,
    ) -> None:
        """Store the encoding name and optional cache dir; load nothing yet."""
        self.encoding = encoding
        self._cache_dir = str(cache_dir) if cache_dir is not None else None
        self._encoder: tiktoken.Encoding | None = None

    def _get_encoder(self) -> tiktoken.Encoding:
        if self._encoder is None:
            if self._cache_dir is not None:
                os.environ["TIKTOKEN_CACHE_DIR"] = self._cache_dir
            import tiktoken

            self._encoder = tiktoken.get_encoding(self.encoding)
        return self._encoder

    def count(self, text: str) -> int:
        """Return ``len(encoding.encode(text))``."""
        if not text:
            return 0
        return len(self._get_encoder().encode(text))


class HeuristicCounter:
    """A rough ``len(text) / 4`` estimate — the keyless, dependency-free floor.

    Under-counts code and CJK by 2–3x (plan.md §3). Constructing one emits a
    warning; use a real tokenizer for anything that feeds a token budget.
    """

    _CHARS_PER_TOKEN = 4

    def __init__(self, *, warn: bool = True) -> None:
        """Warn (unless ``warn=False``) that this counter is only an estimate."""
        if warn:
            warnings.warn(
                "HeuristicCounter estimates tokens as len(text) / 4; it "
                "under-counts code and CJK by 2-3x. Use a real tokenizer for "
                "context budgeting.",
                stacklevel=2,
            )

    def count(self, text: str) -> int:
        """Return the ceiling of ``len(text) / 4`` (0 for empty text)."""
        if not text:
            return 0
        return (len(text) + self._CHARS_PER_TOKEN - 1) // self._CHARS_PER_TOKEN
