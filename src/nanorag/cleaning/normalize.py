"""Whitespace, unicode-form and line-ending normalisation.

Not the loader's job (plan.md §6): a ``Loader`` returns raw extracted text
untouched; :func:`clean_text` is the explicit next pipeline step, before
chunking. It does not touch document structure or strip boilerplate — that
is ``cleaning/boilerplate.py``, Phase G.
"""

from __future__ import annotations

import re

from nanorag.hashing import normalize_text

#: Three or more consecutive newlines (two or more blank lines) collapse to one.
_BLANK_RUN = re.compile(r"\n{3,}")


def clean_text(text: str) -> str:
    r"""Return *text* with tidied whitespace, unicode form and line endings.

    Applies the same NFC + LF transform as
    :func:`nanorag.hashing.normalize_text`, strips trailing whitespace from
    every line, collapses runs of two or more blank lines to exactly one, and
    trims leading/trailing blank lines from the whole document. Pure and
    idempotent: ``clean_text(clean_text(x)) == clean_text(x)``.

    Parameters
    ----------
    text
        Arbitrary text, as returned by a ``Loader``.

    Returns
    -------
    str
        The cleaned text.

    """
    text = normalize_text(text)
    text = "\n".join(line.rstrip() for line in text.split("\n"))
    text = _BLANK_RUN.sub("\n\n", text)
    return text.strip("\n")
