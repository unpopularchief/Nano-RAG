import builtins
from pathlib import Path

import pypdf
import pytest
from pypdf import PdfWriter
from pypdf.generic import (
    DecodedStreamObject,
    DictionaryObject,
    NameObject,
)

from nanorag.chunking import RecursiveChunker
from nanorag.errors import ConfigError, LoaderError
from nanorag.hashing import content_hash, stable_doc_id
from nanorag.loaders import PdfLoader
from nanorag.loaders.pdf import PAGE_SEPARATOR

DATA = Path(__file__).parent / "data"


def _write_text_pdf(path: Path, page_texts: list[str]) -> None:
    writer = PdfWriter()
    font = DictionaryObject(
        {
            NameObject("/Type"): NameObject("/Font"),
            NameObject("/Subtype"): NameObject("/Type1"),
            NameObject("/BaseFont"): NameObject("/Helvetica"),
        }
    )
    font_ref = writer._add_object(font)
    for text in page_texts:
        page = writer.add_blank_page(width=612, height=792)
        page[NameObject("/Resources")] = DictionaryObject(
            {NameObject("/Font"): DictionaryObject({NameObject("/F1"): font_ref})}
        )
        escaped = text.replace("\\", "\\\\").replace("(", "\\(").replace(")", "\\)")
        stream = DecodedStreamObject()
        stream.set_data(f"BT /F1 12 Tf 72 720 Td ({escaped}) Tj ET".encode("latin-1"))
        page[NameObject("/Contents")] = writer._add_object(stream)
    with path.open("wb") as output:
        writer.write(output)


def test_committed_three_page_pdf_loads():
    doc = PdfLoader().load(DATA / "sample.pdf")

    assert doc.metadata == {
        "nanorag.mime": "application/pdf",
        "nanorag.pages": 3,
    }
    assert doc.text.count(PAGE_SEPARATOR) == 2
    assert doc.content_hash == content_hash(doc.text)


def test_pdf_extraction_and_chunk_offsets_address_extracted_text():
    doc = PdfLoader().load(DATA / "sample.pdf", source_uri="fixture.pdf")
    chunks = RecursiveChunker(target_tokens=7, overlap_tokens=1).chunk(doc)

    assert "page one covers loading" in doc.text
    assert "Page two proves extracted text" in doc.text
    assert "Page three verifies exact chunk character offsets" in doc.text
    assert doc.doc_id == stable_doc_id("fixture.pdf")
    assert chunks
    assert all(doc.text[c.start_char : c.end_char] == c.text for c in chunks)


def test_pdf_loader_rejects_encrypted_pdf(tmp_path):
    path = tmp_path / "encrypted.pdf"
    writer = PdfWriter()
    writer.add_blank_page(width=72, height=72)
    writer.encrypt("secret")
    with path.open("wb") as output:
        writer.write(output)

    with pytest.raises(LoaderError, match="encrypted PDF"):
        PdfLoader().load(path)


def test_encrypted_pdf_fails_only_that_file_in_directory_walk(tmp_path):
    from nanorag.loaders import DirectoryLoader

    (tmp_path / "good.txt").write_text("still loaded", encoding="utf-8")
    writer = PdfWriter()
    writer.add_blank_page(width=72, height=72)
    writer.encrypt("secret")
    with (tmp_path / "encrypted.pdf").open("wb") as output:
        writer.write(output)

    report = DirectoryLoader().load_path(tmp_path)

    assert {doc.source_uri for doc in report.loaded} == {"good.txt"}
    assert {issue.source_uri for issue in report.failed} == {"encrypted.pdf"}


def test_pdf_loader_rejects_malformed_pdf(tmp_path):
    path = tmp_path / "broken.pdf"
    path.write_bytes(b"this is not a pdf")

    with pytest.raises(LoaderError, match="cannot parse PDF"):
        PdfLoader().load(path)


def test_pdf_loader_wraps_page_extraction_failure(tmp_path, monkeypatch):
    path = tmp_path / "present.pdf"
    path.write_bytes(b"present so the size guard passes")

    class BrokenPage:
        def extract_text(self):
            raise RuntimeError("broken content stream")

    class BrokenReader:
        is_encrypted = False
        pages = [BrokenPage()]

        def __init__(self, path):
            pass

    monkeypatch.setattr(pypdf, "PdfReader", BrokenReader)

    with pytest.raises(LoaderError, match="cannot extract text from PDF"):
        PdfLoader().load(path)


def test_pdf_loader_enforces_file_page_and_extracted_text_bounds(tmp_path):
    sample = DATA / "sample.pdf"
    with pytest.raises(LoaderError, match="max_file_size"):
        PdfLoader(max_file_size=10).load(sample)
    with pytest.raises(LoaderError, match="max_pages"):
        PdfLoader(max_pages=2).load(sample)

    text_path = tmp_path / "text.pdf"
    _write_text_pdf(text_path, ["more than five characters"])
    with pytest.raises(LoaderError, match="max_extracted_chars"):
        PdfLoader(max_extracted_chars=5).load(text_path)


def test_missing_pdf_extra_names_the_install(monkeypatch):
    real_import = builtins.__import__

    def without_pypdf(name, *args, **kwargs):
        if name == "pypdf":
            raise ImportError("pypdf deliberately hidden")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", without_pypdf)

    with pytest.raises(ConfigError, match=r"nanorag\[pdf\].*pypdf"):
        PdfLoader().load(DATA / "sample.pdf")
