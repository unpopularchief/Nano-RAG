"""Stable, deterministic identifiers for documents and chunks.

Every id in ``nanorag`` is a truncated SHA-256 of text that has first been put
through one shared normalisation (:func:`normalize_text`): NFC unicode form and
LF line endings. Because the hash is computed over explicitly UTF-8-encoded,
normalised bytes — never Python's built-in ``hash()`` — the results are
identical across processes, platforms, interpreter versions and
``PYTHONHASHSEED`` values.

Why ``chunk_id`` hashes the chunk's own text and not the document hash
--------------------------------------------------------------------
Editing one paragraph rotates the ids of only the chunks whose text *or
ordinal* changed. An insertion near the top of a document shifts every
downstream ordinal, so those ids rotate too — but the embedding cache is keyed
by ``(model_id, sha256(normalised chunk text))``, not by ``chunk_id``, so a
chunk whose text is unchanged is still served from cache when its ordinal moved
(plan.md §5).
"""

from __future__ import annotations

import hashlib
import unicodedata

#: Hex length of a document id and a content hash.
DOC_ID_LEN = 32
CONTENT_HASH_LEN = 32
#: Hex length of a chunk id — shorter; collisions are scoped within a document.
CHUNK_ID_LEN = 16

#: NUL is used to separate the fields of a chunk id payload so that no value of
#: ``doc_id``, ``ordinal`` or chunk text can forge a different field boundary.
_FIELD_SEP = "\x00"


def normalize_text(text: str) -> str:
    r"""Return *text* in the canonical form used for all hashing.

    The single shared transform: collapse ``\r\n`` and lone ``\r`` to ``\n``,
    then apply Unicode NFC normalisation. ``content_hash`` and ``chunk_id``
    both call this so an id never changes merely because a file was saved with
    different line endings or a different but canonically equivalent unicode
    encoding.

    Parameters
    ----------
    text
        Arbitrary text.

    Returns
    -------
    str
        The normalised text.

    """
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    return unicodedata.normalize("NFC", text)


def _sha256_hex(data: str, length: int) -> str:
    """SHA-256 of the UTF-8 bytes of *data*, truncated to *length* hex chars."""
    return hashlib.sha256(data.encode("utf-8")).hexdigest()[:length]


def content_hash(text: str) -> str:
    """Return the change-detection hash of a document's text.

    Parameters
    ----------
    text
        The document's full text.

    Returns
    -------
    str
        ``CONTENT_HASH_LEN`` hex characters.

    """
    return _sha256_hex(normalize_text(text), CONTENT_HASH_LEN)


def stable_doc_id(source_uri: str) -> str:
    """Return a stable document id derived from its source URI.

    A document keeps the same id across re-ingests as long as its
    ``source_uri`` is unchanged, so an edited file updates its existing row
    rather than creating a duplicate.

    Parameters
    ----------
    source_uri
        The canonical location of the source (a path, a URL).

    Returns
    -------
    str
        ``DOC_ID_LEN`` hex characters.

    """
    return _sha256_hex(normalize_text(source_uri), DOC_ID_LEN)


def chunk_id(doc_id: str, ordinal: int, text: str) -> str:
    """Return the id of a chunk.

    Derived from the owning ``doc_id``, the chunk's ``ordinal`` within that
    document, and the chunk's *own* normalised text — never the document's
    content hash (see the module docstring).

    Parameters
    ----------
    doc_id
        Id of the document this chunk belongs to.
    ordinal
        Zero-based position of the chunk within the document.
    text
        The chunk's text.

    Returns
    -------
    str
        ``CHUNK_ID_LEN`` hex characters.

    """
    payload = _FIELD_SEP.join((doc_id, str(ordinal), normalize_text(text)))
    return _sha256_hex(payload, CHUNK_ID_LEN)
