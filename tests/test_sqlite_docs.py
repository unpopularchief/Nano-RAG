import sqlite3

import numpy as np
import pytest

from nanorag.errors import IndexModelMismatch, StoreError
from nanorag.hashing import chunk_id, content_hash, stable_doc_id
from nanorag.store.numpy_store import NumpyVectorStore
from nanorag.store.sqlite_docs import SqliteDocumentStore
from nanorag.types import Chunk, Document


def _doc(source_uri: str, text: str) -> Document:
    return Document(
        doc_id=stable_doc_id(source_uri),
        source_uri=source_uri,
        text=text,
        content_hash=content_hash(text),
        metadata={"nanorag.mime": "text/plain"},
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
            )
        )
    return chunks


# --- documents + chunks --------------------------------------------------


def test_roundtrip_document_and_chunks(tmp_path):
    doc = _doc("a.txt", "hello world. goodbye world.")
    chunks = _chunks(doc, "hello world.", "goodbye world.")

    store = SqliteDocumentStore(tmp_path / "db.sqlite3")
    store.upsert_document(doc, chunks)
    store.close()

    reopened = SqliteDocumentStore(tmp_path / "db.sqlite3")
    assert reopened.get_document(doc.doc_id) == doc
    assert reopened.get_chunks(doc.doc_id) == chunks
    reopened.close()


def test_get_document_returns_none_when_missing(tmp_path):
    store = SqliteDocumentStore(tmp_path / "db.sqlite3")
    assert store.get_document("missing") is None


def test_used_as_a_context_manager_closes_on_exit(tmp_path):
    doc = _doc("a.txt", "hello")
    with SqliteDocumentStore(tmp_path / "db.sqlite3") as store:
        store.upsert_document(doc, [])
        assert store.get_document(doc.doc_id) == doc

    with pytest.raises(sqlite3.ProgrammingError):
        store.get_document(doc.doc_id)


def test_upsert_document_replaces_stale_chunks(tmp_path):
    doc = _doc("a.txt", "one two three")
    store = SqliteDocumentStore(tmp_path / "db.sqlite3")
    store.upsert_document(doc, _chunks(doc, "one", "two", "three"))
    assert len(store.get_chunks(doc.doc_id)) == 3

    edited = _doc("a.txt", "one two three")
    store.upsert_document(edited, _chunks(edited, "one two three"))
    remaining = store.get_chunks(doc.doc_id)
    assert len(remaining) == 1
    assert remaining[0].text == "one two three"


def test_upsert_document_preserves_created_at_across_update(tmp_path):
    doc = _doc("a.txt", "hello")
    store = SqliteDocumentStore(tmp_path / "db.sqlite3")
    store.upsert_document(doc, [])
    created_before = store._conn.execute(
        "SELECT created_at FROM documents WHERE doc_id = ?", (doc.doc_id,)
    ).fetchone()[0]

    store.upsert_document(doc, [])
    created_after = store._conn.execute(
        "SELECT created_at FROM documents WHERE doc_id = ?", (doc.doc_id,)
    ).fetchone()[0]
    assert created_before == created_after


def test_upsert_document_rejects_chunk_with_mismatched_doc_id(tmp_path):
    doc = _doc("a.txt", "hello world")
    other = _doc("b.txt", "hello world")
    bad_chunk = _chunks(other, "hello world")[0]

    store = SqliteDocumentStore(tmp_path / "db.sqlite3")
    with pytest.raises(StoreError):
        store.upsert_document(doc, [bad_chunk])


def test_iter_documents_and_iter_chunks(tmp_path):
    store = SqliteDocumentStore(tmp_path / "db.sqlite3")
    doc_a = _doc("a.txt", "aaa")
    doc_b = _doc("b.txt", "bbb")
    store.upsert_document(doc_a, _chunks(doc_a, "aaa"))
    store.upsert_document(doc_b, _chunks(doc_b, "bbb"))

    assert {d.doc_id for d in store.iter_documents()} == {doc_a.doc_id, doc_b.doc_id}
    assert {c.doc_id for c in store.iter_chunks()} == {doc_a.doc_id, doc_b.doc_id}


def test_count_documents_and_chunks(tmp_path):
    store = SqliteDocumentStore(tmp_path / "db.sqlite3")
    assert (store.count_documents(), store.count_chunks()) == (0, 0)
    doc_a = _doc("a.txt", "aa bb")
    doc_b = _doc("b.txt", "cc")
    store.upsert_document(doc_a, _chunks(doc_a, "aa", "bb"))
    store.upsert_document(doc_b, _chunks(doc_b, "cc"))
    assert (store.count_documents(), store.count_chunks()) == (2, 3)
    assert store.count_chunks(doc_a.doc_id) == 2
    assert store.count_chunks("no-such-doc") == 0
    store.delete_document(doc_a.doc_id)
    assert (store.count_documents(), store.count_chunks()) == (1, 1)


def test_delete_document_cascades_to_chunks_and_embeddings(tmp_path):
    doc = _doc("a.txt", "hello world")
    chunks = _chunks(doc, "hello world")
    store = SqliteDocumentStore(tmp_path / "db.sqlite3")
    store.upsert_document(doc, chunks)
    vectors = np.ones((1, 4), dtype=np.float32)
    store.upsert_embeddings("m1", 4, [chunks[0].chunk_id], vectors)

    store.delete_document(doc.doc_id)

    assert store.get_document(doc.doc_id) is None
    assert store.get_chunks(doc.doc_id) == []
    assert list(store.iter_embeddings()) == []


def test_mid_batch_chunk_insert_failure_leaves_no_partial_rows(tmp_path):
    store = SqliteDocumentStore(tmp_path / "db.sqlite3")

    existing = _doc("existing.txt", "shared text")
    existing_chunks = _chunks(existing, "shared text")
    store.upsert_document(existing, existing_chunks)

    colliding_id = existing_chunks[0].chunk_id
    new_doc = _doc("new.txt", "brand new text")
    good_chunk = _chunks(new_doc, "brand new text")[0]
    colliding_chunk = Chunk(
        chunk_id=colliding_id,  # PK collision with `existing`'s chunk
        doc_id=new_doc.doc_id,
        ordinal=1,
        text="second",
        start_char=0,
        end_char=6,
        token_count=1,
    )

    with pytest.raises(sqlite3.IntegrityError):
        store.upsert_document(new_doc, [good_chunk, colliding_chunk])

    assert store.get_document(new_doc.doc_id) is None
    assert store.get_chunks(new_doc.doc_id) == []
    # the pre-existing document is untouched
    assert store.get_chunks(existing.doc_id) == existing_chunks


# --- embeddings -----------------------------------------------------------


def test_index_meta_is_none_before_any_embeddings(tmp_path):
    store = SqliteDocumentStore(tmp_path / "db.sqlite3")
    assert store.index_meta() is None


def test_embeddings_roundtrip_bit_exact(tmp_path):
    doc = _doc("a.txt", "one two")
    chunks = _chunks(doc, "one", "two")
    store = SqliteDocumentStore(tmp_path / "db.sqlite3")
    store.upsert_document(doc, chunks)

    rng = np.random.default_rng(0)
    vectors = rng.standard_normal((2, 8)).astype(np.float32)
    ids = [c.chunk_id for c in chunks]
    store.upsert_embeddings("model-x", 8, ids, vectors)

    assert store.index_meta() == ("model-x", 8)
    roundtripped = dict(store.iter_embeddings())
    for cid, original in zip(ids, vectors, strict=True):
        assert np.array_equal(roundtripped[cid], original)
        assert roundtripped[cid].dtype == np.float32


def test_upsert_embeddings_second_model_raises_index_model_mismatch(tmp_path):
    store = SqliteDocumentStore(tmp_path / "db.sqlite3")
    doc = _doc("a.txt", "hello")
    chunks = _chunks(doc, "hello")
    store.upsert_document(doc, chunks)
    ids = [c.chunk_id for c in chunks]

    store.upsert_embeddings("model-a", 4, ids, np.ones((1, 4), dtype=np.float32))

    with pytest.raises(IndexModelMismatch):
        store.upsert_embeddings("model-b", 4, ids, np.ones((1, 4), dtype=np.float32))


def test_upsert_embeddings_same_model_different_dim_raises_index_model_mismatch(
    tmp_path,
):
    store = SqliteDocumentStore(tmp_path / "db.sqlite3")
    doc = _doc("a.txt", "hello")
    chunks = _chunks(doc, "hello")
    store.upsert_document(doc, chunks)
    ids = [c.chunk_id for c in chunks]

    store.upsert_embeddings("model-a", 4, ids, np.ones((1, 4), dtype=np.float32))

    with pytest.raises(IndexModelMismatch):
        store.upsert_embeddings("model-a", 8, ids, np.ones((1, 8), dtype=np.float32))


def test_upsert_embeddings_rejects_wrong_vector_width(tmp_path):
    store = SqliteDocumentStore(tmp_path / "db.sqlite3")
    doc = _doc("a.txt", "hello")
    chunks = _chunks(doc, "hello")
    store.upsert_document(doc, chunks)

    with pytest.raises(StoreError):
        store.upsert_embeddings(
            "model-a", 4, [chunks[0].chunk_id], np.ones((1, 3), dtype=np.float32)
        )


def test_upsert_embeddings_rejects_length_mismatch(tmp_path):
    store = SqliteDocumentStore(tmp_path / "db.sqlite3")
    doc = _doc("a.txt", "one two")
    chunks = _chunks(doc, "one", "two")
    store.upsert_document(doc, chunks)

    with pytest.raises(StoreError):
        store.upsert_embeddings(
            "model-a",
            4,
            [c.chunk_id for c in chunks],
            np.ones((1, 4), dtype=np.float32),
        )


def test_iter_embeddings_empty_store(tmp_path):
    store = SqliteDocumentStore(tmp_path / "db.sqlite3")
    assert list(store.iter_embeddings()) == []


# --- reopen + rebuild the vector store (B3 checkpoint) ---------------------


def test_reopen_rebuilds_vector_store_and_reproduces_search(tmp_path):
    db_path = tmp_path / "db.sqlite3"
    doc = _doc("a.txt", "cats and dogs and birds")
    chunks = _chunks(doc, "cats", "dogs", "birds")
    rng = np.random.default_rng(42)
    vectors = rng.standard_normal((3, 6)).astype(np.float32)
    vectors /= np.linalg.norm(vectors, axis=1, keepdims=True)

    store = SqliteDocumentStore(db_path)
    store.upsert_document(doc, chunks)
    ids = [c.chunk_id for c in chunks]
    store.upsert_embeddings("model-x", 6, ids, vectors)

    live_index = NumpyVectorStore(dim=6)
    live_index.upsert(ids, vectors)
    query = vectors[1]
    before = live_index.search(query, k=3)
    store.close()

    reopened = SqliteDocumentStore(db_path)
    rebuilt_index = NumpyVectorStore(dim=6)
    for cid, vec in reopened.iter_embeddings():
        rebuilt_index.upsert([cid], vec.reshape(1, -1))
    after = rebuilt_index.search(query, k=3)

    assert before == after


# --- FTS5 / BM25 (plan.md §9 Phase F session F2) ----------------------------


def _requires_fts5(store):
    if not store.fts5_available:
        pytest.skip("FTS5 not available in this SQLite build")


def test_search_bm25_finds_the_matching_chunk(tmp_path):
    store = SqliteDocumentStore(tmp_path / "db.sqlite3")
    _requires_fts5(store)
    doc = _doc("a.txt", "the quick brown fox jumps over the lazy dog")
    chunks = _chunks(doc, "the quick brown fox jumps over the lazy dog")
    store.upsert_document(doc, chunks)

    hits = store.search_bm25('"fox" OR "dog"', 5)
    assert hits and hits[0][0] == chunks[0].chunk_id
    assert hits[0][1] > 0


def test_search_bm25_rejects_k_less_than_one(tmp_path):
    store = SqliteDocumentStore(tmp_path / "db.sqlite3")
    _requires_fts5(store)
    with pytest.raises(StoreError):
        store.search_bm25('"x"', 0)


def test_search_bm25_raises_store_error_when_fts5_unavailable(tmp_path):
    store = SqliteDocumentStore(tmp_path / "db.sqlite3")
    store._fts5_available = False  # noqa: SLF001 - simulate a build without FTS5
    with pytest.raises(StoreError):
        store.search_bm25('"x"', 5)


def test_search_bm25_allowed_ids_is_a_pre_filter_not_a_post_filter(tmp_path):
    store = SqliteDocumentStore(tmp_path / "db.sqlite3")
    _requires_fts5(store)
    doc = _doc("a.txt", "alpha match one. alpha match two. alpha match three.")
    chunks = _chunks(doc, "alpha match one.", "alpha match two.", "alpha match three.")
    store.upsert_document(doc, chunks)
    allowed = {chunks[2].chunk_id}

    hits = store.search_bm25('"alpha"', 1, allowed_ids=allowed)
    assert [cid for cid, _ in hits] == [chunks[2].chunk_id]


def test_search_bm25_allowed_ids_batches_across_the_in_batch_limit(tmp_path):
    import nanorag.store.sqlite_docs as sqlite_docs_module

    store = SqliteDocumentStore(tmp_path / "db.sqlite3")
    _requires_fts5(store)
    n = sqlite_docs_module._IN_BATCH + 5
    doc = _doc("a.txt", " ".join(f"word{i}" for i in range(n)))
    chunks = _chunks(doc, *(f"word{i}" for i in range(n)))
    store.upsert_document(doc, chunks)
    allowed = {c.chunk_id for c in chunks}

    hits = store.search_bm25(
        " OR ".join(f'"word{i}"' for i in range(n)), n, allowed_ids=allowed
    )
    assert {cid for cid, _ in hits} == allowed


def test_chunks_fts_stays_in_sync_across_upsert_and_delete(tmp_path):
    store = SqliteDocumentStore(tmp_path / "db.sqlite3")
    _requires_fts5(store)
    doc = _doc("a.txt", "one two three")
    chunks = _chunks(doc, "one", "two", "three")
    store.upsert_document(doc, chunks)
    assert store.search_bm25('"two"', 5)[0][0] == chunks[1].chunk_id

    # Re-upserting with fewer chunks must drop the stale ones from chunks_fts.
    edited = _doc("a.txt", "one three")
    store.upsert_document(edited, _chunks(edited, "one", "three"))
    assert store.search_bm25('"two"', 5) == []

    # A cascading document delete must also clear chunks_fts (this is what
    # `PRAGMA recursive_triggers = ON` is for: an `AFTER DELETE ON chunks`
    # trigger does not fire for an FK-cascaded delete without it).
    store.delete_document(edited.doc_id)
    assert store.search_bm25('"one"', 5) == []


def test_fts5_backfills_chunks_written_before_it_was_available(tmp_path):
    db_path = tmp_path / "db.sqlite3"
    store = SqliteDocumentStore(db_path)
    _requires_fts5(store)
    doc = _doc("a.txt", "backfilled text")
    chunks = _chunks(doc, "backfilled text")
    store.upsert_document(doc, chunks)

    # Simulate this row having been written while chunks_fts didn't exist
    # (FTS5 unavailable, or the table simply predates this feature) by
    # deleting straight from the FTS5 table without going through `chunks`.
    store._conn.execute(  # noqa: SLF001
        "DELETE FROM chunks_fts WHERE chunk_id = ?", (chunks[0].chunk_id,)
    )
    store._conn.commit()  # noqa: SLF001
    store.close()

    reopened = SqliteDocumentStore(db_path)
    assert reopened.search_bm25('"backfilled"', 5)[0][0] == chunks[0].chunk_id
    reopened.close()
