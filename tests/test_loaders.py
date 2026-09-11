import os
from pathlib import Path

import pytest

from nanorag.errors import LoaderError
from nanorag.hashing import content_hash
from nanorag.loaders import DirectoryLoader, MarkdownLoader, TextLoader
from nanorag.loaders.base import read_text_file

CORPUS = Path(__file__).parent / "data" / "corpus"

# --- TextLoader --------------------------------------------------------


def test_text_loader_reads_plain_utf8():
    doc = TextLoader().load(CORPUS / "plain_utf8.txt")
    assert "café" in doc.text
    assert doc.metadata["nanorag.mime"] == "text/plain"


def test_text_loader_strips_utf8_bom():
    doc = TextLoader().load(CORPUS / "bom_utf8.txt")
    assert not doc.text.startswith(chr(0xFEFF))
    assert "résumé" in doc.text


def test_text_loader_falls_back_to_latin1():
    doc = TextLoader().load(CORPUS / "latin1.txt")
    assert "café" in doc.text
    assert "naïve" in doc.text


def test_text_loader_preserves_crlf_in_raw_text():
    doc = TextLoader().load(CORPUS / "crlf.txt")
    assert "\r\n" in doc.text


def test_text_loader_handles_empty_file():
    doc = TextLoader().load(CORPUS / "empty.txt")
    assert doc.text == ""
    assert doc.content_hash == content_hash("")


def test_text_loader_rejects_nul_bytes():
    with pytest.raises(LoaderError):
        TextLoader().load(CORPUS / "corrupt.txt")


def test_text_loader_rejects_oversized_file():
    with pytest.raises(LoaderError):
        TextLoader(max_file_size=10).load(CORPUS / "plain_utf8.txt")


def test_text_loader_rejects_missing_file(tmp_path):
    with pytest.raises(LoaderError):
        TextLoader().load(tmp_path / "does_not_exist.txt")


def test_text_loader_default_source_uri_is_posix_path():
    doc = TextLoader().load(CORPUS / "plain_utf8.txt")
    assert doc.source_uri == (CORPUS / "plain_utf8.txt").as_posix()


def test_text_loader_explicit_source_uri_overrides_default():
    doc = TextLoader().load(CORPUS / "plain_utf8.txt", source_uri="custom/uri.txt")
    assert doc.source_uri == "custom/uri.txt"


def test_text_loader_doc_id_and_content_hash_are_deterministic():
    a = TextLoader().load(CORPUS / "plain_utf8.txt", source_uri="x.txt")
    b = TextLoader().load(CORPUS / "plain_utf8.txt", source_uri="x.txt")
    assert a.doc_id == b.doc_id
    assert a.content_hash == b.content_hash


# --- MarkdownLoader ------------------------------------------------------


def test_markdown_loader_reads_markdown():
    doc = MarkdownLoader().load(CORPUS / "top.md")
    assert "# Nano RAG" in doc.text
    assert doc.metadata["nanorag.mime"] == "text/markdown"


def test_markdown_loader_rejects_nul_bytes():
    with pytest.raises(LoaderError):
        MarkdownLoader().load(CORPUS / "corrupt.txt")


# --- read_text_file (shared helper) --------------------------------------


def test_read_text_file_prefers_utf8_over_latin1():
    text = read_text_file(CORPUS / "plain_utf8.txt")
    assert isinstance(text, str)


def test_read_text_file_wraps_read_bytes_oserror(tmp_path):
    # stat() succeeds on a directory (it has a size), but read_bytes() must
    # fail — IsADirectoryError on Linux, PermissionError on Windows, both
    # OSError — exercising the read_bytes() error branch specifically,
    # distinct from the stat() branch covered by the missing-file test above.
    with pytest.raises(LoaderError):
        read_text_file(tmp_path)


# --- cross-platform content_hash checkpoint (plan.md §9 Phase B, B1) -----


def test_content_hash_is_identical_regardless_of_line_ending_style():
    crlf_doc = TextLoader().load(CORPUS / "crlf.txt")
    lf_equivalent = crlf_doc.text.replace("\r\n", "\n")
    assert crlf_doc.content_hash == content_hash(lf_equivalent)


# --- DirectoryLoader -------------------------------------------------------


def test_directory_loader_loads_all_recognized_extensions():
    report = DirectoryLoader().load_path(CORPUS)
    loaded_uris = {d.source_uri for d in report.loaded}
    assert "plain_utf8.txt" in loaded_uris
    assert "bom_utf8.txt" in loaded_uris
    assert "latin1.txt" in loaded_uris
    assert "crlf.txt" in loaded_uris
    assert "empty.txt" in loaded_uris
    assert "top.md" in loaded_uris
    assert "nested/deep.txt" in loaded_uris
    assert "ignored_dir/skip_me.txt" in loaded_uris


def test_directory_loader_reports_unrecognized_extension_as_skipped():
    report = DirectoryLoader().load_path(CORPUS)
    skipped_uris = {i.source_uri for i in report.skipped}
    assert "image.png" in skipped_uris
    assert all(d.source_uri != "image.png" for d in report.loaded)


def test_directory_loader_reports_corrupt_file_as_failed_without_aborting():
    report = DirectoryLoader().load_path(CORPUS)
    failed_uris = {i.source_uri for i in report.failed}
    assert "corrupt.txt" in failed_uris
    # a genuinely bad file doesn't stop the rest of the corpus from loading
    assert len(report.loaded) >= 5


def test_directory_loader_glob_filters_by_extension():
    report = DirectoryLoader().load_path(CORPUS, glob="*.md")
    loaded_uris = {d.source_uri for d in report.loaded}
    assert loaded_uris == {"top.md"}
    # non-matching files are excluded from the report entirely, not skipped
    assert all(i.source_uri != "plain_utf8.txt" for i in report.skipped)


def test_directory_loader_ignore_excludes_matching_paths_entirely():
    report = DirectoryLoader().load_path(CORPUS, ignore=["ignored_dir/**"])
    all_uris = (
        {d.source_uri for d in report.loaded}
        | {i.source_uri for i in report.skipped}
        | {i.source_uri for i in report.failed}
    )
    assert "ignored_dir/skip_me.txt" not in all_uris


def test_directory_loader_ignore_excludes_a_matching_top_level_file():
    # A file-only ignore pattern (no directory component) exercises the
    # per-file ignore check directly, distinct from directory-level pruning
    # (the "ignored_dir/**" case above prunes before the file check runs).
    report = DirectoryLoader().load_path(CORPUS, ignore=["plain_utf8.txt"])
    all_uris = (
        {d.source_uri for d in report.loaded}
        | {i.source_uri for i in report.skipped}
        | {i.source_uri for i in report.failed}
    )
    assert "plain_utf8.txt" not in all_uris


def test_directory_loader_source_uri_is_posix_relative_to_root():
    report = DirectoryLoader().load_path(CORPUS)
    for doc in report.loaded:
        assert "\\" not in doc.source_uri
        assert not doc.source_uri.startswith("/")


def test_directory_loader_rejects_non_directory_root():
    with pytest.raises(LoaderError):
        DirectoryLoader().load_path(CORPUS / "plain_utf8.txt")


def test_directory_loader_custom_loader_mapping(tmp_path):
    (tmp_path / "a.txt").write_text("hello", encoding="utf-8")
    report = DirectoryLoader(loaders={}).load_path(tmp_path)
    assert report.loaded == ()
    assert {i.source_uri for i in report.skipped} == {"a.txt"}


def test_directory_loader_skips_symlinked_file(tmp_path):
    target = tmp_path / "real.txt"
    target.write_text("real content", encoding="utf-8")
    link = tmp_path / "link.txt"
    try:
        os.symlink(target, link)
    except OSError:
        pytest.skip("symlink creation not permitted on this platform/user")

    report = DirectoryLoader().load_path(tmp_path)
    loaded_uris = {d.source_uri for d in report.loaded}
    skipped_uris = {i.source_uri for i in report.skipped}
    assert "real.txt" in loaded_uris
    assert "link.txt" in skipped_uris
    assert "link.txt" not in loaded_uris


def test_directory_loader_does_not_follow_symlinked_directory_loop(tmp_path):
    sub = tmp_path / "sub"
    sub.mkdir()
    (sub / "f.txt").write_text("content", encoding="utf-8")
    loop_link = sub / "loop"
    try:
        os.symlink(tmp_path, loop_link, target_is_directory=True)
    except OSError:
        pytest.skip("symlink creation not permitted on this platform/user")

    # must terminate rather than recurse forever through the loop
    report = DirectoryLoader().load_path(tmp_path)
    loaded_uris = {d.source_uri for d in report.loaded}
    assert "sub/f.txt" in loaded_uris
    assert not any(uri.startswith("sub/loop/") for uri in loaded_uris)
