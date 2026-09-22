import builtins
from pathlib import Path

import pytest
import selectolax.parser

from nanorag.chunking import RecursiveChunker
from nanorag.errors import ConfigError, LoaderError
from nanorag.hashing import content_hash, stable_doc_id
from nanorag.loaders import DirectoryLoader, HtmlLoader

DATA = Path(__file__).parent / "data"


def test_committed_html_page_loads_visible_text_only():
    doc = HtmlLoader().load(DATA / "sample.html", source_uri="sample.html")

    assert "HTML loading" in doc.text
    assert "preserving block boundaries" in doc.text
    assert "window.secretNoise" not in doc.text
    assert "color: black" not in doc.text
    assert doc.metadata == {
        "nanorag.mime": "text/html",
        "nanorag.title": "Nano RAG loader fixture",
    }
    assert doc.doc_id == stable_doc_id("sample.html")
    assert doc.content_hash == content_hash(doc.text)


def test_html_chunk_offsets_address_extracted_document_text():
    doc = HtmlLoader().load(DATA / "sample.html")
    chunks = RecursiveChunker(target_tokens=6, overlap_tokens=1).chunk(doc)

    assert chunks
    assert all(doc.text[c.start_char : c.end_char] == c.text for c in chunks)


def test_html_loader_preserves_inline_text_and_block_boundaries(tmp_path):
    path = tmp_path / "inline.html"
    path.write_text(
        "<main><h1>Heading</h1><p>A <strong>bold</strong> claim.</p>"
        "<p>Second paragraph.<br>Next line.</p></main>",
        encoding="utf-8",
    )

    doc = HtmlLoader().load(path)

    assert "A bold claim." in doc.text
    assert "claim.\nSecond paragraph." in doc.text
    assert "Second paragraph.\nNext line." in doc.text


def test_html_loader_handles_a_fragment_without_title(tmp_path):
    path = tmp_path / "fragment.htm"
    path.write_text("<p>Small fragment</p>", encoding="utf-8")

    doc = HtmlLoader().load(path)

    assert doc.text == "Small fragment"
    assert doc.metadata == {"nanorag.mime": "text/html"}


def test_html_loader_rejects_binary_malformed_input(tmp_path):
    path = tmp_path / "broken.html"
    path.write_bytes(b"<html>\x00not text</html>")

    with pytest.raises(LoaderError, match="NUL byte"):
        HtmlLoader().load(path)


def test_html_loader_wraps_parser_failure(tmp_path, monkeypatch):
    path = tmp_path / "broken.html"
    path.write_text("<html>", encoding="utf-8")

    class RootlessParser:
        body = None
        root = None

        def __init__(self, raw):
            pass

        def css(self, selector):
            return []

    monkeypatch.setattr(selectolax.parser, "HTMLParser", RootlessParser)

    with pytest.raises(LoaderError, match="parser produced no document root"):
        HtmlLoader().load(path)


def test_malformed_html_fails_only_that_file_in_directory_walk(tmp_path):
    (tmp_path / "good.html").write_text("<p>Still loaded</p>", encoding="utf-8")
    (tmp_path / "bad.html").write_bytes(b"<html>\x00broken</html>")

    report = DirectoryLoader().load_path(tmp_path)

    assert {doc.source_uri for doc in report.loaded} == {"good.html"}
    assert {issue.source_uri for issue in report.failed} == {"bad.html"}


def test_html_loader_rejects_oversized_file():
    with pytest.raises(LoaderError, match="max_file_size"):
        HtmlLoader(max_file_size=10).load(DATA / "sample.html")


def test_missing_html_extra_names_the_install(monkeypatch):
    real_import = builtins.__import__

    def without_selectolax(name, *args, **kwargs):
        if name == "selectolax.parser":
            raise ImportError("selectolax deliberately hidden")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", without_selectolax)

    with pytest.raises(ConfigError, match=r"nanorag\[html\].*selectolax"):
        HtmlLoader().load(DATA / "sample.html")
