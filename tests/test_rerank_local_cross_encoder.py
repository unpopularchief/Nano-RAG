"""Real-ONNX-model tests for ``LocalCrossEncoderReranker`` (plan.md §9 Phase F).

Opt-in (``-m local``): offline after the first download, but the model
download itself makes this too slow for the default suite (plan.md §12).
Run explicitly with::

    uv run pytest -m local tests/test_rerank_local_cross_encoder.py

There is no smaller substitute model here the way ``test_embeddings_local.py``
uses ``bge-small`` for embeddings — ``Xenova/ms-marco-MiniLM-L-6-v2`` is
already the lightest cross-encoder ``fastembed`` supports (plan.md §4's
"small (~80 MB)" default), so every test uses it directly.
"""

from __future__ import annotations

import pytest

from nanorag.errors import ConfigError, RerankError
from nanorag.hashing import content_hash, stable_doc_id
from nanorag.rerank.base import Reranker
from nanorag.rerank.local_cross_encoder import (
    DEFAULT_MODEL_ID,
    LocalCrossEncoderReranker,
)
from nanorag.types import Chunk, Document, ScoredChunk

pytestmark = pytest.mark.local

_DOC = Document(
    doc_id=stable_doc_id("d.txt"),
    source_uri="d.txt",
    text="text",
    content_hash=content_hash("text"),
    metadata={},
)


def _hit(ordinal: int, text: str) -> ScoredChunk:
    return ScoredChunk(
        chunk=Chunk(
            chunk_id=f"c{ordinal}",
            doc_id=_DOC.doc_id,
            ordinal=ordinal,
            text=text,
            start_char=0,
            end_char=len(text),
            token_count=1,
        ),
        score=0.5,  # the reranker's own score replaces this
        source="dense",
    )


def test_default_model_id_is_the_documented_small_default():
    assert DEFAULT_MODEL_ID == "Xenova/ms-marco-MiniLM-L-6-v2"


def test_local_cross_encoder_satisfies_the_protocol():
    assert isinstance(LocalCrossEncoderReranker(), Reranker)


def test_unknown_model_id_raises_config_error():
    with pytest.raises(ConfigError):
        LocalCrossEncoderReranker("not/a-real-cross-encoder")


def test_rerank_ranks_the_more_relevant_document_first():
    reranker = LocalCrossEncoderReranker()
    hits = [
        _hit(0, "Cats sleep most of the day and purr when content."),
        _hit(1, "The stock market closed lower today amid rate fears."),
    ]

    out = reranker.rerank("Tell me about cats sleeping", hits, top_n=2)

    assert [h.chunk.ordinal for h in out] == [0, 1]
    assert out[0].score > out[1].score
    assert all(h.source == "rerank:local" for h in out)


def test_top_n_slices_the_reranked_list():
    reranker = LocalCrossEncoderReranker()
    hits = [_hit(i, f"document number {i}") for i in range(5)]
    out = reranker.rerank("document number 3", hits, top_n=2)
    assert len(out) == 2


def test_empty_hits_returns_empty():
    reranker = LocalCrossEncoderReranker()
    assert reranker.rerank("anything", [], top_n=5) == []


def test_a_broken_underlying_model_raises_rerank_error(monkeypatch):
    reranker = LocalCrossEncoderReranker()

    def _boom(*args, **kwargs):
        raise RuntimeError("onnxruntime blew up")

    monkeypatch.setattr(reranker._model, "rerank", _boom)
    with pytest.raises(RerankError):
        reranker.rerank("q", [_hit(0, "a")], top_n=1)
