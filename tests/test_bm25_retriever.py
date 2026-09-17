"""``Bm25Retriever``: FTS5 vs. a hand-computed BM25 reference, the NumPy
fallback (the FTS5-unavailable path), pre-filtering, and wiring — plan.md
§9 Phase F session F2."""

import math

import pytest

from nanorag.errors import RetrievalError, StoreError
from nanorag.hashing import chunk_id, content_hash, stable_doc_id
from nanorag.retrieval import Bm25Retriever
from nanorag.retrieval.bm25 import (
    SOURCE,
    _fts5_match_query,
    _tokenize,
    numpy_bm25_scores,
)
from nanorag.store import SqliteDocumentStore
from nanorag.types import Chunk, Document


def _doc(source_uri: str, metadata=None) -> Document:
    text = f"text of {source_uri}"
    return Document(
        doc_id=stable_doc_id(source_uri),
        source_uri=source_uri,
        text=text,
        content_hash=content_hash(text),
        metadata=metadata or {},
    )


def _chunk(doc: Document, ordinal: int, text: str, metadata=None) -> Chunk:
    return Chunk(
        chunk_id=chunk_id(doc.doc_id, ordinal, text),
        doc_id=doc.doc_id,
        ordinal=ordinal,
        text=text,
        start_char=0,
        end_char=len(text),
        token_count=len(text.split()),
        metadata=metadata or {},
    )


@pytest.fixture
def store():
    docs = SqliteDocumentStore(":memory:")
    yield docs
    docs.close()


def _seed(docs: SqliteDocumentStore, texts: list[str], metas=None) -> list[Chunk]:
    doc = _doc("corpus.txt")
    chunks = [
        _chunk(doc, i, t, (metas[i] if metas else None)) for i, t in enumerate(texts)
    ]
    docs.upsert_document(doc, chunks)
    return chunks


# --- construction -------------------------------------------------------------


@pytest.mark.parametrize("k", [0, -1])
def test_non_positive_k_is_rejected_at_construction_and_at_call(store, k):
    with pytest.raises(RetrievalError):
        Bm25Retriever(store, k=k)
    retriever = Bm25Retriever(store)
    with pytest.raises(RetrievalError):
        retriever.retrieve("q", k=k)


def test_default_k_property(store):
    assert Bm25Retriever(store, k=7).k == 7


def test_empty_index_returns_empty_list(store):
    assert Bm25Retriever(store).retrieve("anything") == []


def test_query_with_no_tokens_returns_empty_list_not_an_error(store):
    _seed(store, ["hello world"])
    assert Bm25Retriever(store).retrieve("???") == []
    assert Bm25Retriever(store).retrieve("") == []


# --- the F2 checkpoint: matches a hand-computed BM25 reference ---------------


def _reference_bm25(query: str, texts: list[str], *, k1: float = 1.2, b: float = 0.75):
    """The textbook Okapi BM25 formula, computed by hand from tokens."""
    doc_tokens = [_tokenize(t) for t in texts]
    n = len(texts)
    avg_len = sum(len(toks) for toks in doc_tokens) / n
    query_terms = dict.fromkeys(_tokenize(query))
    scores = [0.0] * n
    for term in query_terms:
        df = sum(1 for toks in doc_tokens if term in toks)
        if df == 0:
            continue
        idf = math.log((n - df + 0.5) / (df + 0.5) + 1.0)
        for i, toks in enumerate(doc_tokens):
            tf = toks.count(term)
            denom = tf + k1 * (1 - b + b * len(toks) / avg_len)
            scores[i] += idf * (tf * (k1 + 1)) / denom
    return scores


TOY_CORPUS = [
    "the quick brown fox jumps over the lazy dog",
    "a quick fox is quick and brown",
    "completely unrelated text about sqlite databases",
    "dogs and foxes rarely interact in the wild",
]
TOY_QUERY = "quick fox"


def test_numpy_fallback_edge_cases():
    from nanorag.hashing import chunk_id as _cid

    def _standalone_chunk(text: str, ordinal: int = 0) -> Chunk:
        return Chunk(
            chunk_id=_cid("d", ordinal, text),
            doc_id="d",
            ordinal=ordinal,
            text=text,
            start_char=0,
            end_char=len(text),
            token_count=len(text.split()),
        )

    assert numpy_bm25_scores("anything", []) == {}
    assert numpy_bm25_scores("???", [_standalone_chunk("hello")]) == {}
    # Every candidate is empty text: avg_len == 0, nothing can match.
    assert numpy_bm25_scores("hello", [_standalone_chunk("")]) == {}
    # A query term absent from every candidate contributes nothing (df=0),
    # but a term that *is* present still scores.
    scores = numpy_bm25_scores(
        "xyzzy hello", [_standalone_chunk("hello world"), _standalone_chunk("goodbye")]
    )
    assert set(scores) == {_standalone_chunk("hello world").chunk_id}


def test_numpy_fallback_matches_the_hand_computed_reference(store):
    chunks = _seed(store, TOY_CORPUS)
    expected = _reference_bm25(TOY_QUERY, TOY_CORPUS)
    got = numpy_bm25_scores(TOY_QUERY, chunks)
    for chunk, exp in zip(chunks, expected, strict=True):
        if exp > 0:
            assert got[chunk.chunk_id] == pytest.approx(exp, rel=1e-9)
        else:
            assert chunk.chunk_id not in got


def test_fts5_path_ranks_consistently_with_the_hand_computed_reference(store):
    if not store.fts5_available:
        pytest.skip("FTS5 not available in this SQLite build")
    chunks = _seed(store, TOY_CORPUS)
    expected = _reference_bm25(TOY_QUERY, TOY_CORPUS)
    expected_order = [
        cid
        for cid, _ in sorted(
            ((c.chunk_id, s) for c, s in zip(chunks, expected, strict=True) if s > 0),
            key=lambda pair: -pair[1],
        )
    ]
    hits = Bm25Retriever(store).retrieve(TOY_QUERY, k=10)
    assert [h.chunk.chunk_id for h in hits] == expected_order
    assert all(h.source == SOURCE for h in hits)


def test_fts5_and_numpy_fallback_agree_on_ranking(store):
    # Different score scales (bm25() vs. the hand-rolled formula's own
    # rounding), but the same corpus must rank the same way under both.
    if not store.fts5_available:
        pytest.skip("FTS5 not available in this SQLite build")
    _seed(store, TOY_CORPUS)
    fts5_order = [h.chunk.chunk_id for h in Bm25Retriever(store).retrieve(TOY_QUERY)]

    store._fts5_available = False  # noqa: SLF001 - force the fallback path
    fallback_hits = Bm25Retriever(store).retrieve(TOY_QUERY)
    fallback_order = [h.chunk.chunk_id for h in fallback_hits]
    assert fts5_order == fallback_order


# --- the FTS5-unavailable path ------------------------------------------------


def test_fts5_unavailable_uses_the_numpy_fallback_and_still_finds_the_match(store):
    _seed(store, TOY_CORPUS)
    store._fts5_available = False  # noqa: SLF001 - simulate a build without FTS5
    retriever = Bm25Retriever(store)
    assert retriever.fts5_available is False
    hits = retriever.retrieve(TOY_QUERY, k=1)
    assert hits[0].chunk.text == "a quick fox is quick and brown"
    assert hits[0].source == SOURCE


def test_search_bm25_raises_when_fts5_unavailable(store):
    store._fts5_available = False  # noqa: SLF001
    with pytest.raises(StoreError):
        store.search_bm25("x", 1)


def test_capability_check_catches_a_build_without_fts5(monkeypatch, tmp_path):
    # sqlite3.Connection is a C type and cannot be monkeypatched directly;
    # instead simulate "the FTS5 module is missing" the way it actually
    # surfaces — the CREATE VIRTUAL TABLE statement itself raising
    # sqlite3.OperationalError — by swapping in a schema that always fails
    # the same way a real missing-module build would.
    import nanorag.store.sqlite_docs as sqlite_docs_module

    monkeypatch.setattr(
        sqlite_docs_module,
        "_FTS_SCHEMA",
        "CREATE VIRTUAL TABLE IF NOT EXISTS chunks_fts USING not_a_real_module(x);",
    )
    docs = SqliteDocumentStore(tmp_path / "db.sqlite3")
    try:
        assert docs.fts5_available is False
    finally:
        docs.close()


# --- pre-filtering -------------------------------------------------------------


def test_filter_matching_nothing_returns_empty_list_not_an_error(store):
    _seed(store, TOY_CORPUS, metas=[{"kind": "a"}] * len(TOY_CORPUS))
    retriever = Bm25Retriever(store)
    assert retriever.retrieve(TOY_QUERY, filter={"kind": "nope"}) == []


def test_filter_restricts_candidates_before_the_top_k_cut(store):
    metas = [{"kind": k} for k in ("wanted", "noise", "noise", "noise")]
    _seed(store, TOY_CORPUS, metas=metas)
    retriever = Bm25Retriever(store)
    hits = retriever.retrieve(TOY_QUERY, k=10, filter={"kind": "wanted"})
    assert len(hits) == 1
    assert hits[0].chunk.metadata["kind"] == "wanted"


def test_filter_works_identically_on_both_scoring_paths(store):
    metas = [{"kind": k} for k in ("wanted", "noise", "noise", "noise")]
    _seed(store, TOY_CORPUS, metas=metas)
    if not store.fts5_available:
        pytest.skip("FTS5 not available in this SQLite build")
    fts5_hits = Bm25Retriever(store).retrieve(
        TOY_QUERY, k=10, filter={"kind": "wanted"}
    )
    store._fts5_available = False  # noqa: SLF001
    fallback_hits = Bm25Retriever(store).retrieve(
        TOY_QUERY, k=10, filter={"kind": "wanted"}
    )
    assert [h.chunk.chunk_id for h in fts5_hits] == [
        h.chunk.chunk_id for h in fallback_hits
    ]


def test_malformed_filter_is_a_retrieval_error(store):
    _seed(store, TOY_CORPUS)
    with pytest.raises(RetrievalError):
        Bm25Retriever(store).retrieve(TOY_QUERY, filter={"kind": {"$regex": "x"}})


# --- misc ----------------------------------------------------------------------


def test_match_query_phrase_quotes_every_term_so_fts5_syntax_is_never_special():
    assert _fts5_match_query("AND OR NOT *") == '"and" OR "or" OR "not"'
    assert _fts5_match_query("   ") is None


def test_hits_carry_the_full_chunk_from_the_document_store(store):
    _seed(store, TOY_CORPUS)
    hit = Bm25Retriever(store).retrieve(TOY_QUERY, k=1)[0]
    by_id = store.get_chunks_by_ids([hit.chunk.chunk_id])
    assert hit.chunk == by_id[hit.chunk.chunk_id]


def test_hit_missing_from_the_document_store_is_a_retrieval_error(store):
    if not store.fts5_available:
        pytest.skip("FTS5 not available in this SQLite build")
    # A chunk_id present in chunks_fts but not in `chunks` itself should
    # never happen in practice (the triggers keep them in sync) but is the
    # same defensive check DenseRetriever makes against its own index.
    store._conn.execute(  # noqa: SLF001
        "INSERT INTO chunks_fts(chunk_id, text) VALUES (?, ?)", ("phantom", "phantom")
    )
    store._conn.commit()  # noqa: SLF001
    with pytest.raises(RetrievalError) as excinfo:
        Bm25Retriever(store).retrieve("phantom")
    assert excinfo.value.context["chunk_id"] == "phantom"
