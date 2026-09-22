"""PDF files into offset-stable extracted-text ``Document`` values.

``pypdf`` is imported only when :meth:`PdfLoader.load` is called. Importing
``nanorag.loaders`` therefore keeps working without the ``[pdf]`` extra, while
trying to load a PDF without it raises an actionable ``ConfigError``.

Offsets produced later by a chunker address the extracted ``Document.text``.
They deliberately do not claim to identify coordinates in the original page
layout: PDF text extraction reflows positioned glyphs (plan.md Phase G1).
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

from nanorag.errors import ConfigError, LoaderError
from nanorag.hashing import content_hash, stable_doc_id
from nanorag.loaders.base import DEFAULT_MAX_FILE_SIZE, _check_file_size
from nanorag.types import Document

if TYPE_CHECKING:
    from pypdf import PdfReader

#: Separates extracted pages without pretending their layout was preserved.
PAGE_SEPARATOR = "\n\f\n"


class PdfLoader:
    """Load a PDF with the optional ``pypdf`` dependency.

    Parameters
    ----------
    max_file_size
        Files larger than this raise ``LoaderError`` before parsing.
    max_pages
        PDFs with more pages than this are rejected before extraction.
    max_extracted_chars
        Maximum extracted text length. This bounds a small compressed input
        that expands to an unexpectedly large in-memory string.

    """

    def __init__(
        self,
        *,
        max_file_size: int = DEFAULT_MAX_FILE_SIZE,
        max_pages: int = 1_000,
        max_extracted_chars: int = DEFAULT_MAX_FILE_SIZE,
    ) -> None:
        """Store resource bounds (see the class docstring)."""
        self.max_file_size = max_file_size
        self.max_pages = max_pages
        self.max_extracted_chars = max_extracted_chars

    def load(self, path: Path, *, source_uri: str | None = None) -> Document:
        """Extract *path* into a ``Document``.

        Parameters
        ----------
        path
            PDF file to parse.
        source_uri
            Canonical location recorded on the document. Defaults to
            ``path.as_posix()``.

        Raises
        ------
        ConfigError
            The ``[pdf]`` extra is not installed.
        LoaderError
            The file is unreadable, encrypted, malformed, or exceeds a
            configured resource bound.

        """
        try:
            from pypdf import PdfReader
        except ImportError as exc:
            raise ConfigError(
                'PDF loading needs the [pdf] extra: run `uv add "nanorag[pdf]"` '
                "(installs pypdf)"
            ) from exc

        _check_file_size(path, max_file_size=self.max_file_size)
        reader = self._open(PdfReader, path)
        if reader.is_encrypted:
            raise LoaderError(f"cannot load encrypted PDF: {path}", path=str(path))
        if len(reader.pages) > self.max_pages:
            raise LoaderError(
                f"{path} has {len(reader.pages)} pages, exceeds max_pages "
                f"({self.max_pages})",
                path=str(path),
                pages=len(reader.pages),
                max_pages=self.max_pages,
            )

        page_texts: list[str] = []
        extracted_chars = 0
        try:
            for page in reader.pages:
                page_text = page.extract_text() or ""
                extracted_chars += len(page_text)
                if extracted_chars > self.max_extracted_chars:
                    raise LoaderError(
                        f"extracted text from {path} exceeds "
                        "max_extracted_chars "
                        f"({self.max_extracted_chars})",
                        path=str(path),
                        extracted_chars=extracted_chars,
                        max_extracted_chars=self.max_extracted_chars,
                    )
                page_texts.append(page_text)
        except LoaderError:
            raise
        except Exception as exc:
            raise LoaderError(
                f"cannot extract text from PDF {path}: {exc}", path=str(path)
            ) from exc

        text = PAGE_SEPARATOR.join(page_texts)
        uri = source_uri if source_uri is not None else path.as_posix()
        return Document(
            doc_id=stable_doc_id(uri),
            source_uri=uri,
            text=text,
            content_hash=content_hash(text),
            metadata={
                "nanorag.mime": "application/pdf",
                "nanorag.pages": len(reader.pages),
            },
        )

    @staticmethod
    def _open(reader_type: type[PdfReader], path: Path) -> PdfReader:
        """Construct a reader and normalise parser failures."""
        try:
            return reader_type(path)
        except Exception as exc:
            raise LoaderError(
                f"cannot parse PDF {path}: {exc}", path=str(path)
            ) from exc
