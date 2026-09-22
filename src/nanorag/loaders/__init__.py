"""Loaders: bytes/path -> ``Document``.

Reached via ``nanorag.loaders``, not the top-level package, so
``import nanorag`` stays free of concrete loader implementations (plan.md §7
"Import cost" — the same rule applied to providers and embedders).
"""

from nanorag.loaders.base import DEFAULT_MAX_FILE_SIZE, Loader
from nanorag.loaders.directory import DEFAULT_LOADERS, DirectoryLoader
from nanorag.loaders.html import HtmlLoader
from nanorag.loaders.markdown import MarkdownLoader
from nanorag.loaders.pdf import PdfLoader
from nanorag.loaders.text import TextLoader

__all__ = [
    "Loader",
    "DEFAULT_MAX_FILE_SIZE",
    "TextLoader",
    "MarkdownLoader",
    "PdfLoader",
    "HtmlLoader",
    "DirectoryLoader",
    "DEFAULT_LOADERS",
]
