import os
import subprocess
import sys
import unicodedata

from nanorag.hashing import (
    CHUNK_ID_LEN,
    CONTENT_HASH_LEN,
    DOC_ID_LEN,
    chunk_id,
    content_hash,
    normalize_text,
    stable_doc_id,
)

# Precomposed "e-acute" (U+00E9) and its canonical decomposition ("e" + U+0301
# COMBINING ACUTE ACCENT). Built from chr()/normalize so the source file stays
# pure ASCII. They render identically but are different strings; NFC folds the
# decomposed form onto the precomposed one.
E_ACUTE_COMPOSED = chr(0x00E9)
E_ACUTE_DECOMPOSED = unicodedata.normalize("NFD", E_ACUTE_COMPOSED)


def test_the_two_e_acute_forms_really_differ():
    assert E_ACUTE_DECOMPOSED != E_ACUTE_COMPOSED
    assert len(E_ACUTE_DECOMPOSED) == 2
    assert len(E_ACUTE_COMPOSED) == 1


def test_normalize_text_collapses_line_endings():
    assert normalize_text("a\r\nb\rc\nd") == "a\nb\nc\nd"


def test_normalize_text_is_idempotent():
    once = normalize_text(E_ACUTE_DECOMPOSED + "\r\ntext\r\n")
    assert normalize_text(once) == once


def test_normalize_text_applies_nfc():
    assert normalize_text(E_ACUTE_DECOMPOSED) == E_ACUTE_COMPOSED
    assert normalize_text(E_ACUTE_DECOMPOSED) == unicodedata.normalize(
        "NFC", E_ACUTE_DECOMPOSED
    )


def test_content_hash_is_deterministic_and_right_length():
    a = content_hash("hello world")
    b = content_hash("hello world")
    assert a == b
    assert len(a) == CONTENT_HASH_LEN
    assert all(c in "0123456789abcdef" for c in a)


def test_content_hash_ignores_line_ending_style():
    assert content_hash("line one\r\nline two") == content_hash("line one\nline two")


def test_content_hash_ignores_unicode_form():
    assert content_hash(E_ACUTE_DECOMPOSED) == content_hash(E_ACUTE_COMPOSED)


def test_content_hash_changes_with_content():
    assert content_hash("a") != content_hash("b")


def test_stable_doc_id_is_deterministic_and_right_length():
    a = stable_doc_id("docs/guide.md")
    assert a == stable_doc_id("docs/guide.md")
    assert len(a) == DOC_ID_LEN


def test_stable_doc_id_distinguishes_sources():
    assert stable_doc_id("docs/a.md") != stable_doc_id("docs/b.md")


def test_chunk_id_length_and_determinism():
    cid = chunk_id("doc1", 0, "some chunk text")
    assert cid == chunk_id("doc1", 0, "some chunk text")
    assert len(cid) == CHUNK_ID_LEN


def test_chunk_id_depends_on_ordinal():
    assert chunk_id("doc1", 0, "text") != chunk_id("doc1", 1, "text")


def test_chunk_id_depends_on_doc_id():
    assert chunk_id("doc1", 0, "text") != chunk_id("doc2", 0, "text")


def test_chunk_id_depends_on_text():
    assert chunk_id("doc1", 0, "text a") != chunk_id("doc1", 0, "text b")


def test_chunk_id_normalises_text():
    assert chunk_id("doc1", 0, "a\r\nb") == chunk_id("doc1", 0, "a\nb")


_CROSS_PROCESS_SCRIPT = r"""
import unicodedata
from nanorag.hashing import chunk_id, content_hash, stable_doc_id

mixed = "mixed " + unicodedata.normalize("NFD", chr(0x00E9)) + " text\r\nend"
print(content_hash(mixed))
print(stable_doc_id("docs/guide.md"))
print(chunk_id("doc-42", 7, mixed))
"""


def _ids_under_hashseed(seed: str) -> str:
    env = {**os.environ, "PYTHONHASHSEED": seed}
    result = subprocess.run(
        [sys.executable, "-c", _CROSS_PROCESS_SCRIPT],
        capture_output=True,
        text=True,
        env=env,
        check=True,
    )
    return result.stdout


def test_ids_are_stable_across_pythonhashseed_values():
    out = _ids_under_hashseed("0")
    assert out == _ids_under_hashseed("1")
    assert len(out.splitlines()) == 3


def test_ids_are_stable_across_repeated_processes():
    assert _ids_under_hashseed("random") == _ids_under_hashseed("random")
