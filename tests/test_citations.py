"""``parse_citations``: marker extraction and resolution against a real
``Context``, the metadata-propagation chain doc -> chunk -> context ->
citation, the "drop unresolvable markers, report a validity rate" contract,
a Hypothesis oracle over arbitrary marker streams, and the D1 checkpoint
(plan.md §9 Phase D: "≥ 95% of markers resolve on the fixture corpus")."""

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from nanorag import Rag
from nanorag.citations import MARKER_PATTERN, CitationReport, parse_citations
from nanorag.context import ContextBuilder
from nanorag.errors import StoreError
from nanorag.hashing import chunk_id, content_hash
from nanorag.store import SqliteDocumentStore
from nanorag.tokens import HeuristicCounter
from nanorag.types import Chunk, Citation, Document, ScoredChunk
from tests.fakes import FakeEmbedder, FakeGenerator

# --- helpers ------------------------------------------------------------------


def _hit(text, *, score=1.0, doc="doc", ordinal=0, start=0, end=None) -> ScoredChunk:
    chunk = Chunk(
        chunk_id=chunk_id(doc, ordinal, text),
        doc_id=doc,
        ordinal=ordinal,
        text=text,
        start_char=start,
        end_char=end if end is not None else start + len(text),
        token_count=len(text.split()),
    )
    return ScoredChunk(chunk=chunk, score=score, source="dense")


def _context(hits, *, budget=10_000):
    return ContextBuilder(budget, counter=HeuristicCounter(warn=False)).build(hits)


def _doc(doc_id, uri):
    return Document(
        doc_id=doc_id, source_uri=uri, text="x" * 100, content_hash=content_hash("x")
    )


def _docs_of(hits, *, uri=lambda doc_id: f"{doc_id}.txt"):
    """A ``get_document`` callable resolving each hit's own ``doc_id``."""
    table = {h.chunk.doc_id: _doc(h.chunk.doc_id, uri(h.chunk.doc_id)) for h in hits}
    return table.get


# --- basic resolution -----------------------------------------------------


def test_single_marker_resolves_with_full_propagation():
    hit = _hit("cats sleep", doc="cats.txt", start=5, end=15)
    context = _context([hit])
    report = parse_citations(
        "Cats sleep a lot. [1]", context, get_document=_docs_of([hit])
    )

    assert report.total_markers == 1
    assert report.resolved_markers == 1
    assert report.validity_rate == 1.0
    assert report.citations == (
        Citation(
            label=1,
            chunk_id=hit.chunk.chunk_id,
            doc_id="cats.txt",
            source_uri="cats.txt.txt",
            start_char=5,
            end_char=15,
        ),
    )


def test_no_markers_is_trivially_fully_valid():
    context = _context([_hit("x")])
    report = parse_citations("I don't know.", context, get_document=lambda d: None)
    assert report == CitationReport(citations=(), total_markers=0, resolved_markers=0)
    assert report.validity_rate == 1.0


def test_multiple_distinct_markers_all_resolve():
    hits = [_hit("a", doc="a"), _hit("b", doc="b"), _hit("c", doc="c")]
    context = _context(hits)
    report = parse_citations(
        "First [1], then [2][3].", context, get_document=_docs_of(hits)
    )
    assert [c.label for c in report.citations] == [1, 2, 3]
    assert report.total_markers == 3
    assert report.resolved_markers == 3


def test_out_of_range_and_zero_markers_are_dropped_but_counted():
    hits = [_hit("a", doc="a")]
    context = _context(hits)
    report = parse_citations(
        "[1] but also [0] and [7].", context, get_document=_docs_of(hits)
    )
    assert [c.label for c in report.citations] == [1]
    assert report.total_markers == 3
    assert report.resolved_markers == 1
    assert report.validity_rate == pytest.approx(1 / 3)


def test_duplicate_marker_collapses_to_one_citation_but_counts_each_occurrence():
    hits = [_hit("a", doc="a")]
    context = _context(hits)
    report = parse_citations(
        "[1] again [1] and again [1].", context, get_document=_docs_of(hits)
    )
    assert len(report.citations) == 1
    assert report.total_markers == 3
    assert report.resolved_markers == 3


def test_out_of_label_order_text_still_returns_citations_in_label_order():
    hits = [_hit("a", doc="a"), _hit("b", doc="b"), _hit("c", doc="c")]
    context = _context(hits)
    report = parse_citations(
        "[3] before [1] before [2].", context, get_document=_docs_of(hits)
    )
    assert [c.label for c in report.citations] == [1, 2, 3]


def test_leading_zero_marker_resolves_by_integer_value():
    hits = [_hit("a", doc="a")]
    context = _context(hits)
    report = parse_citations("[01]", context, get_document=_docs_of(hits))
    assert report.resolved_markers == 1
    assert report.citations[0].label == 1


@pytest.mark.parametrize(
    "text",
    ["[1,2]", "[1-3]", "[abc]", "[]", "no markers here", "[ 1 ]", "【1,2】", "[1】"],
)
def test_non_marker_bracket_shapes_are_not_counted_at_all(text):
    hits = [_hit("a", doc="a")]
    context = _context(hits)
    report = parse_citations(text, context, get_document=_docs_of(hits))
    assert report.total_markers == 0
    assert report.citations == ()


def test_fullwidth_brackets_resolve_like_ascii_ones():
    # openai/gpt-oss-120b writes 【3】 for [3] (D2 live run); both are markers.
    hits = [_hit("a", doc="a"), _hit("b", doc="b"), _hit("c", doc="c")]
    context = _context(hits)
    report = parse_citations(
        "First claim【1】, second [2], hallucinated【9】.",
        context,
        get_document=_docs_of(hits),
    )
    assert (report.total_markers, report.resolved_markers) == (3, 2)
    assert [c.label for c in report.citations] == [1, 2]


def test_missing_document_raises_store_error():
    hit = _hit("a", doc="a")
    context = _context([hit])
    with pytest.raises(StoreError):
        parse_citations("[1]", context, get_document=lambda doc_id: None)


# --- CitationReport value type ----------------------------------------------


def test_citation_report_validates_fields():
    with pytest.raises(TypeError):
        CitationReport(citations=[], total_markers=0, resolved_markers=0)
    with pytest.raises(ValueError):
        CitationReport(citations=(), total_markers=-1, resolved_markers=0)
    with pytest.raises(ValueError):
        CitationReport(citations=(), total_markers=1, resolved_markers=2)


def test_citation_report_to_dict_round_trips_scalars():
    hits = [_hit("a", doc="a")]
    context = _context(hits)
    report = parse_citations("[1]", context, get_document=_docs_of(hits))
    d = report.to_dict()
    assert d["total_markers"] == 1
    assert d["resolved_markers"] == 1
    assert d["validity_rate"] == 1.0
    assert d["citations"][0]["label"] == 1


# --- Hypothesis oracle: parser output matches a reference over any marker stream --


@given(
    n_blocks=st.integers(min_value=1, max_value=12),
    marker_labels=st.lists(st.integers(min_value=0, max_value=20), max_size=30),
)
@settings(max_examples=300, deadline=None)
def test_property_resolution_matches_a_reference_evaluator(n_blocks, marker_labels):
    hits = [_hit(f"chunk {i}", doc=f"doc{i}", ordinal=i) for i in range(n_blocks)]
    context = _context(hits)
    text = "".join(f"[{label}]" for label in marker_labels)
    report = parse_citations(text, context, get_document=_docs_of(hits))

    valid = set(range(1, n_blocks + 1))
    resolved_occurrences = sum(1 for label in marker_labels if label in valid)
    distinct_labels = sorted({label for label in marker_labels if label in valid})

    assert report.total_markers == len(marker_labels)
    assert report.resolved_markers == resolved_occurrences
    assert [c.label for c in report.citations] == distinct_labels
    for citation in report.citations:
        hit = hits[citation.label - 1]
        assert citation.chunk_id == hit.chunk.chunk_id
        assert citation.doc_id == hit.chunk.doc_id
        assert citation.start_char == hit.chunk.start_char
        assert citation.end_char == hit.chunk.end_char
    if marker_labels:
        assert report.validity_rate == pytest.approx(
            resolved_occurrences / len(marker_labels)
        )
    else:
        assert report.validity_rate == 1.0


# --- D1 checkpoint: >= 95% of markers resolve on the fixture corpus ----------

CORPUS = {
    "cats.txt": "Cats sleep most of the day. A cat purrs when content. "
    "Kittens play with string and yarn for hours.",
    "dogs.txt": "Dogs bark at strangers. A dog wags its tail when happy. "
    "Puppies chew on shoes and furniture.",
    "fish.txt": "Fish swim in water. Goldfish live in bowls. "
    "Tropical fish need a heated tank.",
    "birds.txt": "Birds sing at dawn. Parrots can mimic human speech. "
    "Canaries are popular pet songbirds.",
}

QUESTIONS_AND_ANSWERS = [
    # k=3 over a 4-chunk corpus always yields exactly 3 blocks, so [1][2][3]
    # are always valid labels for every query below.
    ("what do cats do", "Cats sleep most of the day and purr when content. [1][2][3]"),
    ("how do dogs behave", "Dogs bark at strangers and wag their tails. [1][2][3]"),
    (
        "what do goldfish need",
        "Goldfish live in bowls; tropical fish need heat. [1][2][3]",
    ),
    ("do birds sing", "Birds sing at dawn and parrots can mimic speech. [1][2][3]"),
    ("kitten behaviour", "Kittens play with string and yarn. [1][2][3]"),
    ("puppy chewing", "Puppies chew on shoes and furniture. [1][2][3]"),
    # One deliberately hallucinated marker among otherwise-valid ones.
    ("mixed sources", "Cats purr and dogs bark. [1][2][3][9]"),
]


def _write_corpus(root):
    for name, body in CORPUS.items():
        (root / name).write_text(body, encoding="utf-8")
    return root


def test_checkpoint_at_least_95_percent_of_markers_resolve_on_the_corpus(tmp_path):
    generator = FakeGenerator([answer for _, answer in QUESTIONS_AND_ANSWERS])
    rag = Rag(
        embedder=FakeEmbedder(dim=128),
        generator=generator,
        docs=SqliteDocumentStore(":memory:"),
        counter=HeuristicCounter(warn=False),
        k=3,
    )
    rag.ingest_path(_write_corpus(tmp_path))

    total_markers = 0
    resolved_markers = 0
    for question, _ in QUESTIONS_AND_ANSWERS:
        answer = rag.query(question)
        # Recompute the same counts the pipeline already logged, from the
        # public Answer alone, to make the checkpoint self-contained.
        resolved_labels = {c.label for c in answer.citations}
        for match in MARKER_PATTERN.finditer(answer.text):
            total_markers += 1
            if int(match.group(1)) in resolved_labels:
                resolved_markers += 1

    # Six answers cite all three retrieved blocks validly (18 markers); the
    # seventh adds one deliberately hallucinated marker ([9]) among three
    # otherwise-valid ones: 21/22 >= 0.95.
    assert total_markers == 22
    assert resolved_markers == 21
    assert resolved_markers / total_markers >= 0.95
