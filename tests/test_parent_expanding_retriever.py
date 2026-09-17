"""``ParentExpandingRetriever``: neighbour-window expansion re-slices the
owning document's text (never concatenates chunk text, which would
double-count chunker overlap), preserves the wrapped retriever's order/
score/source, leaves document edges and single-chunk documents unchanged,
and construction guards — plan.md §9 Phase F session F3."""

import pytest

from nanorag.errors import RetrievalError
from nanorag.hashing import chunk_id, content_hash, stable_doc_id
from nanorag.retrieval.parent import ParentExpandingRetriever
from nanorag.store import SqliteDocumentStore
from nanorag.types import Chunk, Document, ScoredChunk
from tests.fakes import FakeRetriever


def _doc(uri: str, text: str) -> Document:
    return Document(
        doc_id=stable_doc_id(uri),
        source_uri=uri,
        text=text,
        content_hash=content_hash(text),
    )


def _chunks(doc: Document, *texts: str) -> list[Chunk]:
    chunks = []
    pos = 0
    for ordinal, text in enumerate(texts):
        start = doc.text.index(text, pos)
        end = start + len(text)
        pos = end
        chunks.append(
            Chunk(
                chunk_id=chunk_id(doc.doc_id, ordinal, text),
                doc_id=doc.doc_id,
                ordinal=ordinal,
                text=text,
                start_char=start,
                end_char=end,
                token_count=len(text.split()),
                metadata={"tag": f"chunk-{ordinal}"},
            )
        )
    return chunks


@pytest.fixture
def store():
    docs = SqliteDocumentStore(":memory:")
    yield docs
    docs.close()


def _hit(chunk: Chunk, score: float = 0.9, source: str = "dense") -> ScoredChunk:
    return ScoredChunk(chunk=chunk, score=score, source=source)


# --- construction ---------------------------------------------------------


def test_non_positive_window_is_rejected(store):
    with pytest.raises(RetrievalError):
        ParentExpandingRetriever(FakeRetriever([]), store, window=0)


def test_non_positive_k_is_rejected_at_construction_and_at_call(store):
    with pytest.raises(RetrievalError):
        ParentExpandingRetriever(FakeRetriever([]), store, k=0)
    retriever = ParentExpandingRetriever(FakeRetriever([]), store)
    with pytest.raises(RetrievalError):
        retriever.retrieve("q", k=0)


def test_default_k_property(store):
    assert ParentExpandingRetriever(FakeRetriever([]), store, k=7).k == 7


def test_no_hits_returns_empty_list(store):
    assert ParentExpandingRetriever(FakeRetriever([]), store).retrieve("q") == []


# --- the F3 checkpoint: neighbour-window expansion, re-sliced from the doc ----


def test_expands_to_the_window_and_reslices_from_the_document_text(store):
    doc = _doc("a.txt", "alpha. beta. gamma. delta. epsilon.")
    chunks = _chunks(doc, "alpha.", "beta.", "gamma.", "delta.", "epsilon.")
    store.upsert_document(doc, chunks)
    hit = _hit(chunks[2])  # "gamma.", the middle chunk
    retriever = ParentExpandingRetriever(FakeRetriever([hit]), store, window=1)

    [expanded] = retriever.retrieve("q")

    # Re-sliced document.text[start:end], not "beta." + "gamma." + "delta."
    # concatenated — the two are identical here only because there is no
    # chunker overlap in this fixture; the module docstring explains why
    # re-slicing is used regardless.
    assert expanded.chunk.text == "beta. gamma. delta."
    assert expanded.chunk.start_char == chunks[1].start_char
    assert expanded.chunk.end_char == chunks[3].end_char
    assert expanded.chunk.ordinal == chunks[2].ordinal
    assert expanded.chunk.doc_id == doc.doc_id


def test_reslicing_does_not_double_count_chunker_overlap(store):
    # Two chunks that share an overlapping tail/head, as a real chunker
    # with overlap_tokens > 0 would produce. Naive text concatenation would
    # duplicate "gamma"; re-slicing must not.
    text = "alpha beta gamma delta epsilon"
    doc = _doc("a.txt", text)
    c0 = Chunk(
        chunk_id=chunk_id(doc.doc_id, 0, "alpha beta gamma"),
        doc_id=doc.doc_id,
        ordinal=0,
        text="alpha beta gamma",
        start_char=0,
        end_char=17,
        token_count=3,
    )
    c1 = Chunk(
        chunk_id=chunk_id(doc.doc_id, 1, "gamma delta epsilon"),
        doc_id=doc.doc_id,
        ordinal=1,
        text="gamma delta epsilon",  # overlaps c0 on "gamma"
        start_char=12,
        end_char=31,
        token_count=3,
    )
    store.upsert_document(doc, [c0, c1])
    retriever = ParentExpandingRetriever(FakeRetriever([_hit(c0)]), store, window=1)

    [expanded] = retriever.retrieve("q")

    assert expanded.chunk.text == text  # not "alpha beta gammagamma delta epsilon"
    assert expanded.chunk.text.count("gamma") == 1


def test_score_and_source_are_preserved_from_the_wrapped_hit(store):
    doc = _doc("a.txt", "one. two. three.")
    chunks = _chunks(doc, "one.", "two.", "three.")
    store.upsert_document(doc, chunks)
    hit = _hit(chunks[1], score=0.42, source="bm25")
    [expanded] = ParentExpandingRetriever(FakeRetriever([hit]), store).retrieve("q")
    assert expanded.score == 0.42
    assert expanded.source == "bm25"


def test_metadata_is_carried_over_from_the_anchor_chunk(store):
    doc = _doc("a.txt", "one. two. three.")
    chunks = _chunks(doc, "one.", "two.", "three.")
    store.upsert_document(doc, chunks)
    [expanded] = ParentExpandingRetriever(
        FakeRetriever([_hit(chunks[1])]), store
    ).retrieve("q")
    assert expanded.chunk.metadata["tag"] == "chunk-1"


def test_order_and_hit_count_are_preserved_exactly(store):
    doc = _doc("a.txt", "one. two. three. four.")
    chunks = _chunks(doc, "one.", "two.", "three.", "four.")
    store.upsert_document(doc, chunks)
    hits = [_hit(chunks[3]), _hit(chunks[0]), _hit(chunks[2])]  # deliberately unsorted
    retriever = ParentExpandingRetriever(FakeRetriever(hits), store)

    expanded = retriever.retrieve("q", k=3)

    assert len(expanded) == 3
    assert [h.chunk.ordinal for h in expanded] == [3, 0, 2]


# --- edges: document boundary, single-chunk document, store divergence ------


def test_first_chunk_in_a_document_has_no_left_neighbour(store):
    doc = _doc("a.txt", "one. two. three.")
    chunks = _chunks(doc, "one.", "two.", "three.")
    store.upsert_document(doc, chunks)
    [expanded] = ParentExpandingRetriever(
        FakeRetriever([_hit(chunks[0])]), store
    ).retrieve("q")
    assert expanded.chunk.text == "one. two."


def test_last_chunk_in_a_document_has_no_right_neighbour(store):
    doc = _doc("a.txt", "one. two. three.")
    chunks = _chunks(doc, "one.", "two.", "three.")
    store.upsert_document(doc, chunks)
    [expanded] = ParentExpandingRetriever(
        FakeRetriever([_hit(chunks[2])]), store
    ).retrieve("q")
    assert expanded.chunk.text == "two. three."


def test_single_chunk_document_is_returned_unchanged(store):
    doc = _doc("a.txt", "only chunk.")
    chunks = _chunks(doc, "only chunk.")
    store.upsert_document(doc, chunks)
    hit = _hit(chunks[0])
    [result] = ParentExpandingRetriever(FakeRetriever([hit]), store).retrieve("q")
    assert result == hit


def test_wider_window_pulls_in_more_neighbours(store):
    doc = _doc("a.txt", "a. b. c. d. e.")
    chunks = _chunks(doc, "a.", "b.", "c.", "d.", "e.")
    store.upsert_document(doc, chunks)
    [expanded] = ParentExpandingRetriever(
        FakeRetriever([_hit(chunks[2])]), store, window=2
    ).retrieve("q")
    assert expanded.chunk.text == "a. b. c. d. e."


def test_hit_whose_ordinal_is_no_longer_in_the_store_is_returned_unchanged(store):
    # Defensive path: the candidate diverged from the store (should not
    # happen — a retriever only ever returns chunks it just read from the
    # same store), but must degrade rather than raise.
    doc = _doc("a.txt", "one. two.")
    chunks = _chunks(doc, "one.", "two.")
    store.upsert_document(doc, chunks)
    phantom = Chunk(
        chunk_id="phantom",
        doc_id=doc.doc_id,
        ordinal=99,
        text="phantom",
        start_char=0,
        end_char=7,
        token_count=1,
    )
    hit = _hit(phantom)
    [result] = ParentExpandingRetriever(FakeRetriever([hit]), store).retrieve("q")
    assert result == hit


def test_hit_whose_document_is_missing_is_returned_unchanged(store):
    # Same defensive posture for a doc_id with chunks but somehow no
    # document row — constructed directly rather than through the public
    # API, which cannot produce this state.
    doc = _doc("a.txt", "one. two.")
    chunks = _chunks(doc, "one.", "two.")
    store.upsert_document(doc, chunks)
    # Foreign keys off for this one delete, so the document row disappears
    # without cascading away its chunks too — otherwise the earlier
    # "ordinal not in by_ordinal" branch would fire instead of this one.
    store._conn.execute("PRAGMA foreign_keys = OFF")  # noqa: SLF001
    store._conn.execute(  # noqa: SLF001 - force the defensive branch
        "DELETE FROM documents WHERE doc_id = ?", (doc.doc_id,)
    )
    store._conn.commit()  # noqa: SLF001
    hit = _hit(chunks[0])
    [result] = ParentExpandingRetriever(FakeRetriever([hit]), store).retrieve("q")
    assert result == hit


# --- filter pass-through -----------------------------------------------------


def test_filter_and_k_are_forwarded_to_the_wrapped_retriever(store):
    fake = FakeRetriever([])
    ParentExpandingRetriever(fake, store, k=4).retrieve(
        "the question", k=2, filter={"kind": "x"}
    )
    assert fake.calls[0] == ("the question", 2, {"kind": "x"})
