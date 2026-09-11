"""The ``Loader`` protocol and the decode logic shared by its implementations.

A loader's job, precisely (plan.md §6): bytes/path -> ``Document`` with
``source_uri``, raw text and source metadata. Not chunking, not cleaning, not
embedding, not network fetching.
"""

from __future__ import annotations

from pathlib import Path
from typing import Protocol

from nanorag.errors import LoaderError
from nanorag.types import Document

#: Files larger than this raise LoaderError rather than being read into
#: memory whole — the plain-text/markdown equivalent of a zip-bomb guard
#: (plan.md §13 "Resource bounds").
DEFAULT_MAX_FILE_SIZE = 20 * 1024 * 1024  # 20 MiB


class Loader(Protocol):
    """A component that reads one file into a ``Document``.

    Structural, not a base class (plan.md §5 rule 2) — any object with a
    matching ``load`` method satisfies this protocol.
    """

    def load(self, path: Path, *, source_uri: str | None = None) -> Document:
        """Read *path* and return a ``Document``.

        Parameters
        ----------
        path
            File to read.
        source_uri
            Canonical location recorded on the ``Document``. Defaults to
            ``path.as_posix()`` when not given.

        """
        ...


def read_text_file(path: Path, *, max_file_size: int = DEFAULT_MAX_FILE_SIZE) -> str:
    """Decode *path* to text, applying the shared size and content guards.

    Decoding tries ``utf-8-sig`` first (accepts plain UTF-8 and strips a
    BOM), falling back to ``latin-1`` (which cannot fail to decode — every
    byte value is a valid latin-1 code point) so no valid-but-non-UTF-8 text
    file is ever rejected outright.

    Raises
    ------
    LoaderError
        *path* exceeds *max_file_size*, cannot be read (``OSError``), or its
        content contains a NUL byte — a reliable signal the file is binary,
        not text, regardless of extension.

    """
    try:
        size = path.stat().st_size
    except OSError as exc:
        raise LoaderError(f"cannot stat {path}: {exc}", path=str(path)) from exc
    if size > max_file_size:
        raise LoaderError(
            f"{path} is {size} bytes, exceeds max_file_size ({max_file_size})",
            path=str(path),
            size=size,
            max_file_size=max_file_size,
        )

    try:
        raw = path.read_bytes()
    except OSError as exc:
        raise LoaderError(f"cannot read {path}: {exc}", path=str(path)) from exc

    try:
        text = raw.decode("utf-8-sig")
    except UnicodeDecodeError:
        text = raw.decode("latin-1")

    if "\x00" in text:
        raise LoaderError(
            f"{path} contains a NUL byte — not a text file", path=str(path)
        )
    return text
