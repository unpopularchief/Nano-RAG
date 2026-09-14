"""Gate B integration test: the full offline ingest pipeline, at scale.

plan.md §9 Phase B **Acceptance**: "10k chunks ingest and persist with no
network access at all (verified by blocking sockets in the test); reopen +
search reproduces pre-restart results exactly; a second identical ingest
performs zero embedding work." B1-B4 each tested their own component in
isolation; this is the one test that wires all of them together —
``DirectoryLoader`` -> ``FixedChunker`` -> ``FastEmbedEmbedder`` (wrapped in
caching + batching) -> ``SqliteDocumentStore`` + ``NumpyVectorStore`` — the
way a real ingest actually composes them (plan.md §5: "every stage stands
alone", composed by hand since ``Rag`` doesn't exist until Phase C).

Opt-in (``-m local``): needs the real ONNX model. The model is loaded once,
*before* sockets are blocked — the plan's claim is "offline after the first
download", not "never downloads at all", so a from-scratch download inside
the blocked section would test the wrong thing.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from nanorag.chunking.fixed import FixedChunker
from nanorag.embeddings.batching import BatchingEmbedder
from nanorag.embeddings.cache import CachingEmbedder, EmbeddingCache
from nanorag.embeddings.local import FastEmbedEmbedder
from nanorag.loaders.directory import DirectoryLoader
from nanorag.store.numpy_store import NumpyVectorStore
from nanorag.store.sqlite_docs import SqliteDocumentStore
from tests.conftest import blocked_sockets

pytestmark = pytest.mark.local

_MODEL_ID = "BAAI/bge-small-en-v1.5"
_TARGET_CHUNK_COUNT = 10_000
_DOC_COUNT = 200
_PARAGRAPHS_PER_DOC = 46  # calibrated: ~52 chunks/doc at target_tokens=20, ~10.4k total


class _CountingEmbedder:
    """Wraps a real embedder, counting how many texts it actually embeds."""

    def __init__(self, embedder: FastEmbedEmbedder) -> None:
        self._embedder = embedder
        self.dim = embedder.dim
        self.model_id = embedder.model_id
        self.embedded_count = 0

    def embed(self, texts):
        self.embedded_count += len(texts)
        return self._embedder.embed(texts)

    def embed_query(self, text):
        return self._embedder.embed_query(text)


def _write_corpus(root: Path) -> None:
    root.mkdir()
    for i in range(_DOC_COUNT):
        paragraphs = [
            f"Document {i} paragraph {p} covers subject {(i * 7 + p) % 53} "
            f"with some distinguishing detail number {i * 100 + p}."
            for p in range(_PARAGRAPHS_PER_DOC)
        ]
        (root / f"doc_{i:04d}.txt").write_text(
            "\n\n".join(paragraphs), encoding="utf-8"
        )


def test_10k_chunk_ingest_is_offline_reopens_and_reembeds_free(tmp_path):
    # Prime the model cache outside the blocked section (see module docstring).
    FastEmbedEmbedder(_MODEL_ID)

    corpus_dir = tmp_path / "corpus"
    _write_corpus(corpus_dir)
    db_path = tmp_path / "db.sqlite3"
    cache_path = tmp_path / "cache.sqlite3"

    with blocked_sockets():
        report = DirectoryLoader().load_path(corpus_dir)
        assert not report.failed
        assert not report.skipped
        assert len(report.loaded) == _DOC_COUNT

        chunker = FixedChunker(target_tokens=20, overlap_tokens=0)
        docs_and_chunks = [(doc, chunker.chunk(doc)) for doc in report.loaded]
        all_chunks = [c for _, chunks in docs_and_chunks for c in chunks]
        assert len(all_chunks) >= _TARGET_CHUNK_COUNT

        doc_store = SqliteDocumentStore(db_path)
        try:
            for doc, chunks in docs_and_chunks:
                doc_store.upsert_document(doc, chunks)

            embedding_cache = EmbeddingCache(cache_path)
            try:
                embedder = CachingEmbedder(
                    BatchingEmbedder(FastEmbedEmbedder(_MODEL_ID), batch_size=256),
                    embedding_cache,
                )
                texts = [c.text for c in all_chunks]
                ids = [c.chunk_id for c in all_chunks]
                vectors = embedder.embed(texts)
                doc_store.upsert_embeddings(
                    embedder.model_id, embedder.dim, ids, vectors
                )
            finally:
                embedding_cache.close()

            vector_store = NumpyVectorStore(dim=embedder.dim)
            vector_store.upsert(ids, vectors)

            query_vector = vectors[0]
            before = vector_store.search(query_vector, k=5)
        finally:
            doc_store.close()

    # Reopen: still offline — proves the durable path needs no re-embedding.
    with blocked_sockets():
        reopened = SqliteDocumentStore(db_path)
        try:
            rebuilt_store = NumpyVectorStore(dim=embedder.dim)
            for chunk_id, vector in reopened.iter_embeddings():
                rebuilt_store.upsert([chunk_id], vector.reshape(1, -1))
            after = rebuilt_store.search(query_vector, k=5)
            assert before == after
        finally:
            reopened.close()

        # A second, identical ingest re-embeds nothing.
        second_cache = EmbeddingCache(cache_path)
        try:
            counting = _CountingEmbedder(FastEmbedEmbedder(_MODEL_ID))
            second_run = CachingEmbedder(
                BatchingEmbedder(counting, batch_size=256), second_cache
            )
            second_vectors = second_run.embed(texts)
            assert counting.embedded_count == 0
            assert np.array_equal(second_vectors, vectors)
        finally:
            second_cache.close()
