"""``DenseRetriever``: the C1 checkpoint (matches a brute-force NumPy reference
exactly), the Phase C filter tests (matching nothing -> empty, not an error;
ties deterministic; post-filtering provably loses results), and wiring."""

import numpy as np
import pytest

from nanorag.errors import RetrievalError
from nanorag.hashing import chunk_id, content_hash, stable_doc_id
from nanorag.retrieval import DenseRetriever
from nanorag.retrieval.dense import SOURCE
from nanorag.store import NumpyVectorStore, SqliteDocumentStore
from nanorag.types import Chunk, Document
from tests.fakes import FakeEmbedder


class VectorEmbedder:
    """An ``Embedder`` whose query vector is chosen by the test."""

    def __init__(self, dim: int, query: np.ndarray | None = None) -> None:
        self.dim = dim
        self.model_id = "test-vectors"
        self.query = query

    def embed(self, texts):  # pragma: no cover - not used by the retriever
        raise NotImplementedError

    def embed_query(self, text: str) -> np.ndarray:
        assert self.query is not None
        return self.query


def _unit(rows: np.ndarray) -> np.ndarray:
    rows = np.asarray(rows, dtype=np.float32)
    return rows / np.linalg.norm(rows, axis=-1, keepdims=True)


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


def _build(vectors_by_chunk, docs_and_chunks, dim):
    """Persist docs/chunks to SQLite and their vectors to a NumpyVectorStore."""
    docs = SqliteDocumentStore(":memory:")
    for doc, chunks in docs_and_chunks:
        docs.upsert_document(doc, chunks)
    vectors = NumpyVectorStore(dim=dim)
    ids = list(vectors_by_chunk)
    vectors.upsert(ids, np.stack([vectors_by_chunk[i] for i in ids]))
    docs.upsert_embeddings(
        "test-vectors", dim, ids, np.stack([vectors_by_chunk[i] for i in ids])
    )
    return docs, vectors


# --- construction -------------------------------------------------------------


def test_dimension_mismatch_is_a_retrieval_error():
    docs = SqliteDocumentStore(":memory:")
    with pytest.raises(RetrievalError):
        DenseRetriever(FakeEmbedder(dim=8), NumpyVectorStore(dim=4), docs)
    docs.close()


@pytest.mark.parametrize("k", [0, -1])
def test_non_positive_k_is_rejected_at_construction_and_at_call(k):
    docs = SqliteDocumentStore(":memory:")
    with pytest.raises(RetrievalError):
        DenseRetriever(FakeEmbedder(dim=4), NumpyVectorStore(dim=4), docs, k=k)
    retriever = DenseRetriever(FakeEmbedder(dim=4), NumpyVectorStore(dim=4), docs)
    with pytest.raises(RetrievalError):
        retriever.retrieve("q", k=k)
    docs.close()


def test_default_k_property():
    docs = SqliteDocumentStore(":memory:")
    retriever = DenseRetriever(FakeEmbedder(dim=4), NumpyVectorStore(dim=4), docs, k=7)
    assert retriever.k == 7
    docs.close()


# --- the C1 checkpoint: matches a brute-force NumPy reference exactly --------


def _reference(matrix, ids, query, k, allowed=None):
    """The obvious implementation: score everything, sort by (-score, id)."""
    scores = matrix @ query
    rows = [
        (float(scores[i]), ids[i])
        for i in range(len(ids))
        if allowed is None or ids[i] in allowed
    ]
    rows.sort(key=lambda r: (-r[0], r[1]))
    return [(cid, score) for score, cid in rows[:k]]


@pytest.fixture
def random_corpus():
    rng = np.random.default_rng(1234)
    dim, n = 32, 600
    matrix = _unit(rng.standard_normal((n, dim)))
    docs_and_chunks = []
    vectors_by_chunk = {}
    ids = []
    for d in range(30):
        doc = _doc(f"{'even' if d % 2 == 0 else 'odd'}/{d}.txt", {"parity": d % 2})
        chunks = []
        for o in range(20):
            row = d * 20 + o
            c = _chunk(doc, o, f"chunk {row}", {"tens": row // 10})
            chunks.append(c)
            vectors_by_chunk[c.chunk_id] = matrix[row]
            ids.append(c.chunk_id)
        docs_and_chunks.append((doc, chunks))
    docs, vectors = _build(vectors_by_chunk, docs_and_chunks, dim)
    query = _unit(rng.standard_normal(dim))
    yield docs, vectors, matrix, ids, query
    docs.close()


def test_unfiltered_retrieve_matches_brute_force_reference(random_corpus):
    docs, vectors, matrix, ids, query = random_corpus
    retriever = DenseRetriever(VectorEmbedder(32, query), vectors, docs, k=25)
    hits = retriever.retrieve("ignored")
    expected = _reference(matrix, ids, query, k=25)
    assert [(h.chunk.chunk_id, h.score) for h in hits] == [
        (cid, pytest.approx(score, abs=1e-6)) for cid, score in expected
    ]
    assert all(h.source == SOURCE for h in hits)


def test_filtered_retrieve_matches_brute_force_reference_over_the_subset(random_corpus):
    docs, vectors, matrix, ids, query = random_corpus
    retriever = DenseRetriever(VectorEmbedder(32, query), vectors, docs)
    filter = {"parity": 1, "tens": {"$gte": 30}}
    hits = retriever.retrieve("ignored", k=15, filter=filter)
    allowed = docs.filter_chunk_ids(filter)
    assert 0 < len(allowed) < len(ids)
    expected = _reference(matrix, ids, query, k=15, allowed=allowed)
    assert [(h.chunk.chunk_id, h.score) for h in hits] == [
        (cid, pytest.approx(score, abs=1e-6)) for cid, score in expected
    ]
    assert all(h.chunk.metadata["tens"] >= 30 for h in hits)


def test_hits_carry_the_full_chunk_from_the_document_store(random_corpus):
    docs, vectors, matrix, ids, query = random_corpus
    retriever = DenseRetriever(VectorEmbedder(32, query), vectors, docs)
    hit = retriever.retrieve("ignored", k=1)[0]
    assert hit.chunk == docs.get_chunks_by_ids([hit.chunk.chunk_id])[hit.chunk.chunk_id]
    assert hit.chunk.text.startswith("chunk ")


# --- filters ----------------------------------------------------------------


def test_filter_matching_nothing_returns_empty_list_not_an_error(random_corpus):
    docs, vectors, matrix, ids, query = random_corpus
    retriever = DenseRetriever(VectorEmbedder(32, query), vectors, docs)
    assert retriever.retrieve("ignored", filter={"parity": 7}) == []
    assert retriever.retrieve("ignored", filter={"source_uri": "nope.txt"}) == []


def test_empty_filter_is_the_same_as_no_filter(random_corpus):
    docs, vectors, matrix, ids, query = random_corpus
    retriever = DenseRetriever(VectorEmbedder(32, query), vectors, docs, k=5)
    assert retriever.retrieve("ignored", filter={}) == retriever.retrieve("ignored")


def test_malformed_filter_is_a_retrieval_error(random_corpus):
    docs, vectors, matrix, ids, query = random_corpus
    retriever = DenseRetriever(VectorEmbedder(32, query), vectors, docs)
    with pytest.raises(RetrievalError):
        retriever.retrieve("ignored", filter={"parity": {"$regex": "x"}})


def test_post_filtering_silently_loses_results_pre_filtering_does_not():
    # Why pre-filtering is mandatory (plan.md §15 #7). Ten "noise" chunks
    # sit closest to the query but fail the filter; the three chunks that
    # pass the filter are further away. Post-filtering the unfiltered top-3
    # keeps nothing; the pre-filtered top-3 returns all three.
    dim = 4
    query = _unit(np.array([1, 0, 0, 0]))
    noise_doc = _doc("noise.txt", {"kind": "noise"})
    wanted_doc = _doc("wanted.txt", {"kind": "wanted"})
    vectors_by_chunk = {}
    noise_chunks, wanted_chunks = [], []
    for i in range(10):
        c = _chunk(noise_doc, i, f"noise {i}")
        noise_chunks.append(c)
        vectors_by_chunk[c.chunk_id] = _unit(np.array([1, 0.01 * i, 0, 0]))
    for i in range(3):
        c = _chunk(wanted_doc, i, f"wanted {i}")
        wanted_chunks.append(c)
        vectors_by_chunk[c.chunk_id] = _unit(np.array([1, 1, 0.1 * i, 0]))
    docs, vectors = _build(
        vectors_by_chunk, [(noise_doc, noise_chunks), (wanted_doc, wanted_chunks)], dim
    )
    retriever = DenseRetriever(VectorEmbedder(dim, query), vectors, docs)
    k = 3
    wanted = {"kind": "wanted"}

    unfiltered = retriever.retrieve("q", k=k)
    post_filtered = [h for h in unfiltered if h.chunk.metadata.get("kind") == "wanted"]
    assert post_filtered == []  # every wanted chunk was outside the top-k

    pre_filtered = retriever.retrieve("q", k=k, filter=wanted)
    assert sorted(h.chunk.text for h in pre_filtered) == [
        "wanted 0",
        "wanted 1",
        "wanted 2",
    ]
    docs.close()


def test_document_level_metadata_filters_that_documents_chunks():
    # The filter joins to `documents`, so metadata set only on the Document
    # (as loaders do with nanorag.mime) selects all of its chunks.
    dim = 2
    md = _doc("a.md", {"nanorag.mime": "text/markdown"})
    txt = _doc("b.txt", {"nanorag.mime": "text/plain"})
    md_chunks = [_chunk(md, 0, "md chunk")]
    txt_chunks = [_chunk(txt, 0, "txt chunk")]
    v = _unit(np.array([1, 0]))
    docs, vectors = _build(
        {md_chunks[0].chunk_id: v, txt_chunks[0].chunk_id: v},
        [(md, md_chunks), (txt, txt_chunks)],
        dim,
    )
    retriever = DenseRetriever(VectorEmbedder(dim, v), vectors, docs)
    hits = retriever.retrieve("q", filter={"nanorag.mime": "text/markdown"})
    assert [h.chunk.text for h in hits] == ["md chunk"]
    docs.close()


# --- ties, reopen, consistency ------------------------------------------------


def test_ties_are_deterministic_and_identical_after_a_reopen(tmp_path):
    # Identical vectors -> identical scores. Row order in the live store is
    # insertion order; after a reopen it is chunk_id order. The ranking must
    # be the same in both.
    dim = 2
    same = _unit(np.array([0, 1]))
    doc = _doc("dup.txt")
    chunks = [_chunk(doc, i, f"dup {i}") for i in range(6)]
    ids = [c.chunk_id for c in chunks]
    shuffled = [ids[i] for i in (4, 1, 5, 0, 2, 3)]

    docs = SqliteDocumentStore(tmp_path / "db.sqlite3")
    docs.upsert_document(doc, chunks)
    docs.upsert_embeddings("test-vectors", dim, shuffled, np.stack([same] * 6))
    live = NumpyVectorStore(dim=dim)
    live.upsert(shuffled, np.stack([same] * 6))
    retriever = DenseRetriever(VectorEmbedder(dim, same), live, docs, k=3)
    before = [h.chunk.chunk_id for h in retriever.retrieve("q")]
    docs.close()

    reopened_docs = SqliteDocumentStore(tmp_path / "db.sqlite3")
    rebuilt = NumpyVectorStore(dim=dim)
    for cid, vec in reopened_docs.iter_embeddings():
        rebuilt.upsert([cid], vec[None, :])
    retriever = DenseRetriever(VectorEmbedder(dim, same), rebuilt, reopened_docs, k=3)
    after = [h.chunk.chunk_id for h in retriever.retrieve("q")]
    reopened_docs.close()

    assert before == after == sorted(ids)[:3]


def test_hit_missing_from_the_document_store_is_a_retrieval_error():
    dim = 2
    v = _unit(np.array([1, 0]))
    docs = SqliteDocumentStore(":memory:")
    vectors = NumpyVectorStore(dim=dim)
    vectors.upsert(["phantom"], v[None, :])
    retriever = DenseRetriever(VectorEmbedder(dim, v), vectors, docs)
    with pytest.raises(RetrievalError) as excinfo:
        retriever.retrieve("q")
    assert excinfo.value.context["chunk_id"] == "phantom"
    docs.close()


def test_empty_index_returns_empty_list():
    docs = SqliteDocumentStore(":memory:")
    retriever = DenseRetriever(FakeEmbedder(dim=8), NumpyVectorStore(dim=8), docs)
    assert retriever.retrieve("anything") == []
    docs.close()


def test_end_to_end_with_the_fake_embedder_finds_the_topical_chunk():
    embedder = FakeEmbedder(dim=64)
    doc = _doc("notes.txt")
    texts = [
        "the retriever applies metadata filters before search",
        "sqlite stores the chunk text and offsets",
        "numpy holds the vector matrix in memory",
    ]
    chunks = [_chunk(doc, i, t) for i, t in enumerate(texts)]
    ids = [c.chunk_id for c in chunks]
    docs = SqliteDocumentStore(":memory:")
    docs.upsert_document(doc, chunks)
    vectors = NumpyVectorStore(dim=64)
    vectors.upsert(ids, embedder.embed(texts))
    retriever = DenseRetriever(embedder, vectors, docs, k=1)
    [hit] = retriever.retrieve("how does the retriever filter?")
    assert hit.chunk.text == texts[0]
    assert hit.source == "dense"
    docs.close()
