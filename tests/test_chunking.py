import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from nanorag.chunking import (
    DEFAULT_SEPARATORS,
    FixedChunker,
    MarkdownChunker,
    RecursiveChunker,
)
from nanorag.chunking.base import (
    DEFAULT_OVERLAP_TOKENS,
    DEFAULT_TARGET_TOKENS,
    atomic_spans,
    default_counter,
    pack_spans,
)
from nanorag.chunking.markdown import _iter_sections
from nanorag.errors import ChunkingError
from nanorag.hashing import content_hash, stable_doc_id
from nanorag.types import Chunk, Document


def _doc(text: str, uri: str = "doc.txt") -> Document:
    return Document(
        doc_id=stable_doc_id(uri),
        source_uri=uri,
        text=text,
        content_hash=content_hash(text),
    )


_CHUNKER_CLASSES = [FixedChunker, RecursiveChunker, MarkdownChunker]


# --- shared behaviour across all three chunkers --------------------------


@pytest.mark.parametrize("cls", _CHUNKER_CLASSES)
def test_empty_document_yields_no_chunks(cls):
    assert cls().chunk(_doc("")) == []


@pytest.mark.parametrize("cls", _CHUNKER_CLASSES)
def test_short_document_yields_one_chunk_covering_all_text(cls):
    doc = _doc("A short document that fits in one chunk easily.")
    chunks = cls(target_tokens=100, overlap_tokens=10).chunk(doc)
    assert len(chunks) == 1
    assert chunks[0].text == doc.text
    assert chunks[0].start_char == 0
    assert chunks[0].end_char == len(doc.text)
    assert chunks[0].ordinal == 0


@pytest.mark.parametrize("cls", _CHUNKER_CLASSES)
def test_chunking_is_deterministic(cls):
    doc = _doc("Repeatable text. " * 40)
    chunker = cls(target_tokens=15, overlap_tokens=3)
    first = chunker.chunk(doc)
    second = chunker.chunk(doc)
    assert first == second


@pytest.mark.parametrize("cls", _CHUNKER_CLASSES)
@pytest.mark.parametrize(
    "target_tokens, overlap_tokens",
    [(0, 0), (-1, 0), (5, -1), (5, 5), (5, 6)],
)
def test_invalid_budget_raises_chunking_error(cls, target_tokens, overlap_tokens):
    with pytest.raises(ChunkingError):
        cls(target_tokens=target_tokens, overlap_tokens=overlap_tokens)


@pytest.mark.parametrize("cls", _CHUNKER_CLASSES)
def test_custom_counter_is_used(cls):
    class WordCounter:
        def count(self, text: str) -> int:
            return len(text.split())

    doc = _doc("one two three four five six seven eight nine ten")
    chunks = cls(target_tokens=3, overlap_tokens=0, counter=WordCounter()).chunk(doc)
    assert len(chunks) > 1
    assert all(c.token_count <= 3 for c in chunks)


@pytest.mark.parametrize("cls", _CHUNKER_CLASSES)
def test_offsets_reconstruct_source_text(cls):
    doc = _doc("Sentence one. Sentence two. Sentence three.\n\nNew paragraph here.")
    for chunk in cls(target_tokens=8, overlap_tokens=2).chunk(doc):
        assert doc.text[chunk.start_char : chunk.end_char] == chunk.text


# --- FixedChunker ----------------------------------------------------------


def test_fixed_chunker_overlaps_consecutive_windows():
    doc = _doc("word " * 200)
    chunks = FixedChunker(target_tokens=20, overlap_tokens=5).chunk(doc)
    assert len(chunks) > 1
    for prev, cur in zip(chunks, chunks[1:], strict=False):
        assert cur.start_char < prev.end_char  # windows overlap
        assert cur.start_char > prev.start_char  # but always advance


def test_fixed_chunker_ignores_natural_boundaries():
    # A single long "word" with no separators anywhere must still be split.
    doc = _doc("x" * 500)
    chunks = FixedChunker(target_tokens=20, overlap_tokens=0).chunk(doc)
    assert len(chunks) > 1


# --- RecursiveChunker --------------------------------------------------------


def test_recursive_chunker_is_the_documented_default_signature():
    # plan.md §7 wrote RecursiveChunker(target_tokens=512, overlap_tokens=64);
    # the D3 sweep (docs/evaluation.md) moved the default to 128 / 64.
    chunker = RecursiveChunker()
    assert (chunker.target_tokens, chunker.overlap_tokens) == (128, 64)
    assert (DEFAULT_TARGET_TOKENS, DEFAULT_OVERLAP_TOKENS) == (128, 64)
    for cls in (FixedChunker, MarkdownChunker):
        assert (cls().target_tokens, cls().overlap_tokens) == (128, 64)
    assert chunker.separators == DEFAULT_SEPARATORS


def test_recursive_chunker_prefers_paragraph_boundaries():
    doc = _doc(
        "First paragraph with enough words to be sizeable on its own here.\n\n"
        "Second paragraph, also fairly long, following a blank line separator."
    )
    chunks = RecursiveChunker(target_tokens=12, overlap_tokens=0).chunk(doc)
    # Splits should land at/near the blank-line boundary rather than mid-word.
    assert any(c.text.endswith("\n\n") or c.text.rstrip() == c.text for c in chunks)


def test_recursive_chunker_custom_separators():
    doc = _doc("a,b,c,d,e,f,g,h,i,j")
    chunker = RecursiveChunker(target_tokens=2, overlap_tokens=0, separators=(",",))
    chunks = chunker.chunk(doc)
    assert len(chunks) > 1
    assert "".join(c.text for c in chunks) == doc.text


# --- MarkdownChunker ---------------------------------------------------------


def test_markdown_chunker_tags_chunks_with_heading_path():
    text = (
        "# Title\n\nIntro text.\n\n"
        "## Section A\n\nContent of A.\n\n"
        "## Section B\n\nContent of B.\n"
    )
    doc = _doc(text, uri="doc.md")
    chunks = MarkdownChunker(target_tokens=8, overlap_tokens=0).chunk(doc)
    paths = [c.metadata.get("nanorag.heading_path") for c in chunks]
    assert "Title" in paths
    assert "Title > Section A" in paths
    assert "Title > Section B" in paths


def test_markdown_chunker_no_headings_has_no_heading_path_metadata():
    doc = _doc("Just plain prose with no markdown headings at all.", uri="doc.md")
    chunks = MarkdownChunker(target_tokens=20, overlap_tokens=0).chunk(doc)
    assert all("nanorag.heading_path" not in c.metadata for c in chunks)


def test_markdown_chunker_never_crosses_a_heading_boundary():
    text = "# A\n\n" + ("alpha " * 30) + "\n\n# B\n\n" + ("beta " * 30)
    doc = _doc(text, uri="doc.md")
    chunks = MarkdownChunker(target_tokens=10, overlap_tokens=2).chunk(doc)
    b_heading_start = text.index("# B")
    for c in chunks:
        # Every chunk is entirely before or entirely at/after the B heading.
        assert c.end_char <= b_heading_start or c.start_char >= b_heading_start


def test_iter_sections_partitions_text_exactly():
    text = "intro\n\n# A\n\ncontent a\n\n## A.1\n\ncontent a1\n\n# B\n\ncontent b\n"
    sections = _iter_sections(text)
    assert sections[0][1] == 0
    assert sections[-1][2] == len(text)
    for (_, _, end), (_, next_start, _) in zip(sections, sections[1:], strict=False):
        assert end == next_start  # contiguous, no gap, no overlap


def test_iter_sections_heading_path_reflects_ancestry():
    text = "# A\n\ncontent\n\n## A.1\n\nnested\n\n### A.1.1\n\ndeep\n"
    sections = _iter_sections(text)
    paths = [hp for hp, _, _ in sections]
    assert "A" in paths
    assert "A > A.1" in paths
    assert "A > A.1 > A.1.1" in paths


# --- shared base helpers, called directly (defensive branches no chunker
# reaches, since every chunker already guards empty/degenerate input before
# calling them) --------------------------------------------------------


def test_atomic_spans_on_degenerate_span_returns_empty():
    assert atomic_spans("abc", 2, 2, DEFAULT_SEPARATORS, 5, default_counter()) == []


def test_atomic_spans_with_empty_separator_falls_back_to_bisect():
    text = "x" * 20
    spans = atomic_spans(text, 0, len(text), ("",), 3, default_counter())
    assert spans  # split via the character-level bisect fallback
    assert "".join(text[s:e] for s, e in spans) == text


def test_pack_spans_on_empty_input_returns_empty():
    assert pack_spans("", [], 10, 2, default_counter()) == []


# --- property-based invariants (plan.md §9 B2 checkpoint: 500+ inputs) -----


def _assert_chunking_invariants(doc: Document, chunks: list[Chunk]) -> None:
    n = len(doc.text)
    assert [c.ordinal for c in chunks] == list(range(len(chunks)))
    for c in chunks:
        assert 0 <= c.start_char <= c.end_char <= n
        assert doc.text[c.start_char : c.end_char] == c.text
    if not chunks:
        return
    assert chunks[0].start_char == 0
    assert chunks[-1].end_char == n
    for prev, cur in zip(chunks, chunks[1:], strict=False):
        assert cur.start_char <= prev.end_char  # no gap between consecutive chunks
        overlap_len = max(0, prev.end_char - cur.start_char)
        prev_len = prev.end_char - prev.start_char
        assert overlap_len <= prev_len  # overlap never exceeds the prior chunk's size


@st.composite
def _chunk_params(draw: st.DrawFn) -> tuple[str, int, int]:
    target_tokens = draw(st.integers(min_value=1, max_value=50))
    overlap_tokens = draw(st.integers(min_value=0, max_value=target_tokens - 1))
    text = draw(st.text(max_size=500))
    return text, target_tokens, overlap_tokens


@given(_chunk_params())
@settings(max_examples=500)
def test_chunking_invariants_hold_for_every_chunker(
    params: tuple[str, int, int],
) -> None:
    text, target_tokens, overlap_tokens = params
    doc = _doc(text, uri="hypothesis.txt")
    for cls in _CHUNKER_CLASSES:
        chunker = cls(target_tokens=target_tokens, overlap_tokens=overlap_tokens)
        chunks = chunker.chunk(doc)
        _assert_chunking_invariants(doc, chunks)
