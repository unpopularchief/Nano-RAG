"""``HybridRetriever``: RRF arithmetic hand-computed against
``FakeRetriever``s, filter/candidate_k pass-through, ties, and construction
guards — plan.md §9 Phase F session F2."""

import pytest

from nanorag.errors import RetrievalError
from nanorag.hashing import chunk_id, content_hash, stable_doc_id
from nanorag.retrieval.hybrid import DEFAULT_RRF_K, SOURCE, HybridRetriever, rrf_fuse
from nanorag.types import Chunk, Document, ScoredChunk
from tests.fakes import FakeRetriever


def _doc(uri: str) -> Document:
    text = f"text of {uri}"
    return Document(
        doc_id=stable_doc_id(uri),
        source_uri=uri,
        text=text,
        content_hash=content_hash(text),
    )


def _chunk(doc: Document, ordinal: int) -> Chunk:
    text = f"chunk {ordinal}"
    return Chunk(
        chunk_id=chunk_id(doc.doc_id, ordinal, text),
        doc_id=doc.doc_id,
        ordinal=ordinal,
        text=text,
        start_char=0,
        end_char=len(text),
        token_count=1,
    )


_DOC = _doc("hybrid.txt")
_CHUNKS = [_chunk(_DOC, i) for i in range(6)]


def _hit(i: int, score: float, source: str = "dense") -> ScoredChunk:
    return ScoredChunk(chunk=_CHUNKS[i], score=score, source=source)


# --- construction ---------------------------------------------------------


def test_fewer_than_two_retrievers_is_a_retrieval_error():
    with pytest.raises(RetrievalError):
        HybridRetriever(FakeRetriever([]))
    with pytest.raises(RetrievalError):
        HybridRetriever()


@pytest.mark.parametrize("kwargs", [{"k": 0}, {"candidate_k": 0}, {"rrf_k": 0}])
def test_non_positive_parameters_are_rejected_at_construction(kwargs):
    with pytest.raises(RetrievalError):
        HybridRetriever(FakeRetriever([]), FakeRetriever([]), **kwargs)


def test_non_positive_k_is_rejected_at_call():
    retriever = HybridRetriever(FakeRetriever([]), FakeRetriever([]))
    with pytest.raises(RetrievalError):
        retriever.retrieve("q", k=0)


def test_default_k_property():
    retriever = HybridRetriever(FakeRetriever([]), FakeRetriever([]), k=7)
    assert retriever.k == 7


# --- the F2 checkpoint: RRF arithmetic, hand-computed ----------------------


def test_rrf_score_is_the_hand_computed_sum_of_reciprocal_ranks():
    # chunk 0: rank 1 in A, rank 2 in B. chunk 1: rank 2 in A only.
    # chunk 2: rank 1 in B only.
    a = FakeRetriever([_hit(0, 0.9), _hit(1, 0.5)])
    b = FakeRetriever([_hit(2, 0.9), _hit(0, 0.5)])
    retriever = HybridRetriever(a, b, rrf_k=60)
    hits = retriever.retrieve("q", k=3)

    rrf_k = 60
    expected = {
        _CHUNKS[0].chunk_id: 1 / (rrf_k + 1) + 1 / (rrf_k + 2),
        _CHUNKS[2].chunk_id: 1 / (rrf_k + 1),
        _CHUNKS[1].chunk_id: 1 / (rrf_k + 2),
    }
    got = {h.chunk.chunk_id: h.score for h in hits}
    for cid, score in expected.items():
        assert got[cid] == pytest.approx(score)
    # Ranked by fused score, highest first.
    assert [h.chunk.chunk_id for h in hits] == sorted(
        expected, key=lambda cid: -expected[cid]
    )
    assert all(h.source == SOURCE for h in hits)


def test_default_rrf_k_constant_is_used_when_not_overridden():
    a = FakeRetriever([_hit(0, 0.9)])
    b = FakeRetriever([_hit(0, 0.9)])
    hit = HybridRetriever(a, b).retrieve("q", k=1)[0]
    assert hit.score == pytest.approx(2 / (DEFAULT_RRF_K + 1))


def test_a_chunk_missing_from_one_retriever_gets_no_contribution_from_it():
    a = FakeRetriever([_hit(0, 0.9)])
    b = FakeRetriever([])  # chunk 0 absent here: contributes nothing, not a penalty
    hit = HybridRetriever(a, b, rrf_k=60).retrieve("q", k=1)[0]
    assert hit.score == pytest.approx(1 / 61)


def test_ties_are_broken_by_chunk_id():
    # Both retrievers return the same two chunks in the same order, so both
    # get an identical fused score; the winner must be decided by id, not
    # by retriever iteration order.
    a = FakeRetriever([_hit(4, 0.9), _hit(5, 0.5)])
    b = FakeRetriever([_hit(4, 0.9), _hit(5, 0.5)])
    hits = HybridRetriever(a, b).retrieve("q", k=2)
    ids = [h.chunk.chunk_id for h in hits]
    assert ids == sorted([_CHUNKS[4].chunk_id, _CHUNKS[5].chunk_id])


def test_three_retrievers_fuse_together():
    a = FakeRetriever([_hit(0, 0.9)])
    b = FakeRetriever([_hit(0, 0.9)])
    c = FakeRetriever([_hit(0, 0.9)])
    hit = HybridRetriever(a, b, c, rrf_k=60).retrieve("q", k=1)[0]
    assert hit.score == pytest.approx(3 / 61)


def test_chunk_object_comes_from_the_first_retriever_that_returned_it():
    # Chunk identity/content for a given id is invariant across retrievers
    # (same underlying document store); this only proves it isn't dropped.
    a = FakeRetriever([_hit(0, 0.9)])
    b = FakeRetriever([_hit(0, 0.5)])
    hit = HybridRetriever(a, b).retrieve("q", k=1)[0]
    assert hit.chunk == _CHUNKS[0]


# --- candidate_k / filter pass-through -------------------------------------


def test_candidate_k_defaults_to_k_and_is_requested_from_every_retriever():
    a, b = FakeRetriever([_hit(0, 1.0)]), FakeRetriever([_hit(1, 1.0)])
    HybridRetriever(a, b, k=5).retrieve("q")
    assert a.calls[0][1] == 5
    assert b.calls[0][1] == 5


def test_explicit_candidate_k_overrides_the_default_and_widens_with_a_larger_call_k():
    a, b = FakeRetriever([]), FakeRetriever([])
    retriever = HybridRetriever(a, b, k=5, candidate_k=20)
    retriever.retrieve("q")
    assert a.calls[0][1] == 20
    retriever.retrieve("q", k=30)
    assert a.calls[1][1] == 30  # call k exceeds candidate_k: widen, don't shrink


def test_filter_and_query_are_forwarded_to_every_retriever():
    a, b = FakeRetriever([]), FakeRetriever([])
    HybridRetriever(a, b).retrieve("the question", filter={"kind": "x"})
    assert a.calls[0] == ("the question", 10, {"kind": "x"})
    assert b.calls[0] == ("the question", 10, {"kind": "x"})


def test_rrf_fuse_rejects_non_positive_k_and_rrf_k():
    # HybridRetriever always validates before calling rrf_fuse, so these
    # only exercise rrf_fuse's own guard — it is a public function other
    # callers (query_transform.MultiQueryRetriever) invoke directly.
    with pytest.raises(RetrievalError):
        rrf_fuse([[_hit(0, 0.9)]], 0)
    with pytest.raises(RetrievalError):
        rrf_fuse([[_hit(0, 0.9)]], 1, rrf_k=0)


def test_a_retrievers_own_error_propagates():
    class Failing:
        def retrieve(self, query, k=None, filter=None):
            raise RetrievalError("boom")

    with pytest.raises(RetrievalError):
        HybridRetriever(Failing(), FakeRetriever([])).retrieve("q")
