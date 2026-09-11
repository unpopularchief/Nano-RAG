"""``MarkdownLoader`` — markdown files into ``Document``.

Loading is identical to plain text at this phase: markdown-aware structure
(heading paths, code fences) is a *chunker* concern (``chunking/markdown.py``,
Phase B2), not a loader concern (plan.md §6). The only difference here is the
``mime`` recorded in metadata.
"""

from __future__ import annotations

from pathlib import Path

from nanorag.hashing import content_hash, stable_doc_id
from nanorag.loaders.base import DEFAULT_MAX_FILE_SIZE, read_text_file
from nanorag.types import Document


class MarkdownLoader:
    """Loads a markdown file (``.md``, ``.markdown``) into a ``Document``.

    Parameters
    ----------
    max_file_size
        Files larger than this raise ``LoaderError`` instead of being read
        into memory whole.

    """

    def __init__(self, *, max_file_size: int = DEFAULT_MAX_FILE_SIZE) -> None:
        """Store *max_file_size* (see the class docstring)."""
        self.max_file_size = max_file_size

    def load(self, path: Path, *, source_uri: str | None = None) -> Document:
        """Read *path* and return a ``Document`` with ``mime = text/markdown``.

        Parameters
        ----------
        path
            File to read.
        source_uri
            Canonical location recorded on the ``Document``. Defaults to
            ``path.as_posix()``.

        Raises
        ------
        LoaderError
            The file is too large, unreadable, or contains a NUL byte.

        """
        uri = source_uri if source_uri is not None else path.as_posix()
        text = read_text_file(path, max_file_size=self.max_file_size)
        return Document(
            doc_id=stable_doc_id(uri),
            source_uri=uri,
            text=text,
            content_hash=content_hash(text),
            metadata={"nanorag.mime": "text/markdown"},
        )
