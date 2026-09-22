"""HTML files into visible, offset-stable extracted-text ``Document`` values.

The optional ``selectolax`` parser is imported only at load time. The loader
extracts text; it does not perform semantic rewriting or site-specific
boilerplate removal. That remains the explicit next cleaning step.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import TYPE_CHECKING

from nanorag.errors import ConfigError, LoaderError
from nanorag.hashing import content_hash, stable_doc_id
from nanorag.loaders.base import DEFAULT_MAX_FILE_SIZE, read_text_file
from nanorag.types import Document

if TYPE_CHECKING:
    from selectolax.parser import Node

_NON_CONTENT_SELECTOR = "script, style, noscript, template, svg, canvas"
_BLOCK_TAGS = frozenset(
    {
        "address",
        "article",
        "aside",
        "blockquote",
        "br",
        "dd",
        "div",
        "dl",
        "dt",
        "figcaption",
        "figure",
        "footer",
        "form",
        "h1",
        "h2",
        "h3",
        "h4",
        "h5",
        "h6",
        "header",
        "hr",
        "li",
        "main",
        "nav",
        "ol",
        "p",
        "pre",
        "section",
        "table",
        "tbody",
        "td",
        "th",
        "thead",
        "tr",
        "ul",
    }
)
_HORIZONTAL_SPACE = re.compile(r"[^\S\n]+")
_EXCESS_NEWLINES = re.compile(r"\n{3,}")


class HtmlLoader:
    """Load an HTML file with the optional ``selectolax`` dependency.

    Parameters
    ----------
    max_file_size
        Files larger than this raise ``LoaderError`` before parsing.

    """

    def __init__(self, *, max_file_size: int = DEFAULT_MAX_FILE_SIZE) -> None:
        """Store *max_file_size* (see the class docstring)."""
        self.max_file_size = max_file_size

    def load(self, path: Path, *, source_uri: str | None = None) -> Document:
        """Extract visible text from *path* into a ``Document``.

        Parameters
        ----------
        path
            HTML file to parse.
        source_uri
            Canonical location recorded on the document. Defaults to
            ``path.as_posix()``.

        Raises
        ------
        ConfigError
            The ``[html]`` extra is not installed.
        LoaderError
            The file is unreadable, binary, or the parser fails.

        """
        try:
            from selectolax.parser import HTMLParser
        except ImportError as exc:
            raise ConfigError(
                'HTML loading needs the [html] extra: run `uv add "nanorag[html]"` '
                "(installs selectolax)"
            ) from exc

        raw = read_text_file(path, max_file_size=self.max_file_size)
        try:
            tree = HTMLParser(raw)
            for node in tree.css(_NON_CONTENT_SELECTOR):
                node.decompose()
            root = tree.body if tree.body is not None else tree.root
            if root is None:
                raise ValueError("parser produced no document root")
            text = _extract_text(root)
            title_node = tree.css_first("title")
            title = title_node.text(strip=True) if title_node is not None else ""
        except Exception as exc:
            raise LoaderError(
                f"cannot parse HTML {path}: {exc}", path=str(path)
            ) from exc

        uri = source_uri if source_uri is not None else path.as_posix()
        metadata: dict[str, str] = {"nanorag.mime": "text/html"}
        if title:
            metadata["nanorag.title"] = title
        return Document(
            doc_id=stable_doc_id(uri),
            source_uri=uri,
            text=text,
            content_hash=content_hash(text),
            metadata=metadata,
        )


def _extract_text(root: Node) -> str:
    """Return visible text with HTML block boundaries represented as lines."""
    parts: list[str] = []
    for node in root.traverse(include_text=True):
        if node.tag == "-text":
            parts.append(node.text(deep=False))
        elif node.tag == "br":
            parts.append("\n")
        elif node.tag in _BLOCK_TAGS:
            parts.append("\n")
    text = "".join(parts).replace("\r\n", "\n").replace("\r", "\n")
    text = _HORIZONTAL_SPACE.sub(" ", text)
    text = "\n".join(line.strip() for line in text.splitlines())
    return _EXCESS_NEWLINES.sub("\n\n", text).strip()
