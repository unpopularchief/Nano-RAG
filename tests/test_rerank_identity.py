"""``IdentityReranker``: a pure slice, no re-scoring (plan.md §6)."""

from __future__ import annotations

from nanorag.hashing import content_hash, stable_doc_id
from nanorag.rerank.base import Reranker
from nanorag.rerank.identity import IdentityReranker
from nanorag.types import Chunk, Document, ScoredChunk


def _hit(ordinal: int, score: float) -> ScoredChunk:
    doc = Document(
        doc_id=stable_doc_id("d.txt"),
        source_uri="d.txt",
        text="text",
        content_hash=content_hash("text"),
        metadata={},
    )
    return ScoredChunk(
        chunk=Chunk(
            chunk_id=f"c{ordinal}",
            doc_id=doc.doc_id,
            ordinal=ordinal,
            text="t",
            start_char=0,
            end_char=1,
            token_count=1,
        ),
        score=score,
        source="dense",
    )


def test_identity_reranker_satisfies_the_protocol():
    assert isinstance(IdentityReranker(), Reranker)


def test_returns_the_first_top_n_hits_unchanged():
    hits = [_hit(0, 0.9), _hit(1, 0.8), _hit(2, 0.7)]
    result = IdentityReranker().rerank("query", hits, top_n=2)
    assert result == hits[:2]


def test_top_n_larger_than_the_candidate_list_returns_everything():
    hits = [_hit(0, 0.9)]
    assert IdentityReranker().rerank("query", hits, top_n=5) == hits


def test_top_n_zero_returns_an_empty_list():
    hits = [_hit(0, 0.9)]
    assert IdentityReranker().rerank("query", hits, top_n=0) == []


def test_empty_candidate_list_returns_empty():
    assert IdentityReranker().rerank("query", [], top_n=5) == []


def test_query_is_ignored():
    hits = [_hit(0, 0.9), _hit(1, 0.8)]
    reranker = IdentityReranker()
    assert reranker.rerank("a", hits, top_n=2) == reranker.rerank("b", hits, top_n=2)
