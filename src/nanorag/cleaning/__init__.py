"""Text cleaning: pure ``str -> str`` transforms applied before chunking."""

from nanorag.cleaning.normalize import clean_text

__all__ = ["clean_text"]
