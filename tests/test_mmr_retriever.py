"""``MmrRetriever``: the F3 "MMR diversity bound" checkpoint (a near-duplicate
loses to a diverse-but-less-relevant candidate, hand-computed), the
degenerate ``lambda_mult`` ends (pure relevance, pure diversity), tie-break,
missing-vector handling, and construction guards — plan.md §9 Phase F
session F3."""

import numpy as np
import pytest

from nanorag.errors import RetrievalError
from nanorag.retrieval.mmr import SOURCE, MmrRetriever
from nanorag.store.numpy_store import NumpyVectorStore
from nanorag.types import Chunk, ScoredChunk
from tests.fakes import FakeRetriever


def _unit(vec: list[float]) -> np.ndarray:
    arr = np.array(vec, dtype=np.float32)
    return arr / np.linalg.norm(arr)


def _chunk(cid: str, text: str = "x") -> Chunk:
    return Chunk(
        chunk_id=cid,
        doc_id="doc",
        ordinal=0,
        text=text,
        start_char=0,
        end_char=len(text),
        token_count=1,
    )


def _hit(cid: str, score: float) -> ScoredChunk:
    return ScoredChunk(chunk=_chunk(cid), score=score, source="dense")


# --- construction ---------------------------------------------------------


@pytest.mark.parametrize("kwargs", [{"k": 0}, {"candidate_k": 0}])
def test_non_positive_k_parameters_are_rejected(kwargs):
    vectors = NumpyVectorStore(dim=2)
    with pytest.raises(RetrievalError):
        MmrRetriever(FakeRetriever([]), vectors, **kwargs)


@pytest.mark.parametrize("lambda_mult", [-0.01, 1.01])
def test_lambda_mult_out_of_range_is_rejected(lambda_mult):
    vectors = NumpyVectorStore(dim=2)
    with pytest.raises(RetrievalError):
        MmrRetriever(FakeRetriever([]), vectors, lambda_mult=lambda_mult)


def test_non_positive_k_is_rejected_at_call():
    vectors = NumpyVectorStore(dim=2)
    retriever = MmrRetriever(FakeRetriever([]), vectors)
    with pytest.raises(RetrievalError):
        retriever.retrieve("q", k=0)


def test_default_k_property():
    vectors = NumpyVectorStore(dim=2)
    assert MmrRetriever(FakeRetriever([]), vectors, k=7).k == 7


def test_empty_candidates_returns_empty_list():
    vectors = NumpyVectorStore(dim=2)
    retriever = MmrRetriever(FakeRetriever([]), vectors)
    assert retriever.retrieve("q") == []


# --- the F3 checkpoint: MMR diversity bound, hand-computed ------------------


@pytest.fixture
def diversity_scenario():
    """C1, C2 are near-duplicate vectors; C3 is orthogonal to both.

    Plain top-2 by relevance picks C1+C2 (cosine ~0.99 apart). MMR at
    lambda_mult=0.5 must pick C1+C3 instead (cosine 0.0 apart) — hand
    computed below, not just asserted to "look diverse".
    """
    vectors = NumpyVectorStore(dim=2)
    c1, c2, c3 = _unit([1, 0]), _unit([0.99, np.sqrt(1 - 0.99**2)]), _unit([0, 1])
    vectors.upsert(["c1", "c2", "c3"], np.stack([c1, c2, c3]))
    hits = [_hit("c1", 0.9), _hit("c2", 0.85), _hit("c3", 0.5)]
    retriever = FakeRetriever(hits)
    return retriever, vectors, (c1, c2, c3)


def test_naive_top_k_would_pick_the_near_duplicate(diversity_scenario):
    retriever, _, _ = diversity_scenario
    naive = retriever.retrieve("q", k=2)
    assert [h.chunk.chunk_id for h in naive] == ["c1", "c2"]


def test_mmr_picks_the_diverse_candidate_over_the_near_duplicate(diversity_scenario):
    retriever, vectors, (c1, c2, c3) = diversity_scenario
    # candidate_k=3 so all three candidates are in the pool MMR chooses
    # from — MMR can only diversify among what it was actually handed.
    mmr = MmrRetriever(retriever, vectors, k=2, candidate_k=3, lambda_mult=0.5)

    hits = mmr.retrieve("q", k=2)

    assert [h.chunk.chunk_id for h in hits] == ["c1", "c3"]
    assert all(h.source == SOURCE for h in hits)
    # The hand-computed bound: MMR's picks are strictly less mutually
    # similar than the naive top-k's picks.
    naive_pairwise = float(c1 @ c2)
    mmr_pairwise = float(c1 @ c3)
    assert mmr_pairwise < naive_pairwise
    assert mmr_pairwise == pytest.approx(0.0, abs=1e-6)


def test_mmr_scores_match_the_hand_computed_arithmetic(diversity_scenario):
    retriever, vectors, (c1, c2, c3) = diversity_scenario
    mmr = MmrRetriever(retriever, vectors, k=2, candidate_k=3, lambda_mult=0.5)
    hits = mmr.retrieve("q", k=2)

    # relevance normalised min-max over [0.9, 0.85, 0.5] -> lo=0.5, hi=0.9
    rel_c1 = (0.9 - 0.5) / 0.4  # == 1.0
    # first pick (c1): no diversity term yet -> 0.5*rel_norm(c1) - 0.5*0
    assert hits[0].score == pytest.approx(0.5 * rel_c1)
    # second pick (c3): 0.5*rel_norm(c3) - 0.5*cos(c3, c1); rel_norm(c3) = 0, cos = 0
    assert hits[1].score == pytest.approx(0.0, abs=1e-6)


# --- degenerate lambda_mult ends --------------------------------------------


def test_lambda_mult_one_ignores_diversity_and_matches_relevance_order(
    diversity_scenario,
):
    retriever, vectors, _ = diversity_scenario
    mmr = MmrRetriever(retriever, vectors, k=3, lambda_mult=1.0)
    hits = mmr.retrieve("q", k=3)
    assert [h.chunk.chunk_id for h in hits] == ["c1", "c2", "c3"]


def test_lambda_mult_zero_first_pick_is_tie_broken_by_id_not_relevance():
    # With no selection yet, every candidate's diversity penalty is 0, so
    # lambda_mult=0 makes every mmr_score exactly 0.0 for the first pick —
    # the tie-break (chunk id) decides, not relevance.
    vectors = NumpyVectorStore(dim=2)
    vectors.upsert(
        ["a", "b", "c"],
        np.stack([_unit([1, 0]), _unit([0, 1]), _unit([1, 1])]),
    )
    hits = [_hit("c", 0.9), _hit("b", 0.5), _hit("a", 0.1)]  # id order != score order
    mmr = MmrRetriever(
        FakeRetriever(hits), vectors, k=1, candidate_k=3, lambda_mult=0.0
    )
    [only] = mmr.retrieve("q", k=1)
    assert only.chunk.chunk_id == "a"


# --- missing vectors, flat scores -------------------------------------------


def test_candidate_missing_from_the_vector_store_is_not_excluded():
    vectors = NumpyVectorStore(dim=2)
    vectors.upsert(["a"], np.stack([_unit([1, 0])]))
    hits = [_hit("a", 0.9), _hit("phantom", 0.8)]
    mmr = MmrRetriever(FakeRetriever(hits), vectors, k=2)
    ids = {h.chunk.chunk_id for h in mmr.retrieve("q", k=2)}
    assert ids == {"a", "phantom"}


def test_flat_relevance_scores_normalize_to_all_equally_relevant():
    vectors = NumpyVectorStore(dim=2)
    vectors.upsert(["a", "b"], np.stack([_unit([1, 0]), _unit([0, 1])]))
    hits = [_hit("a", 0.5), _hit("b", 0.5)]
    mmr = MmrRetriever(
        FakeRetriever(hits), vectors, k=1, candidate_k=2, lambda_mult=1.0
    )
    [only] = mmr.retrieve("q", k=1)
    assert only.score == pytest.approx(1.0)  # both map to the all-ones case


# --- candidate_k / filter pass-through --------------------------------------


def test_candidate_k_defaults_to_k_and_is_requested_from_the_wrapped_retriever():
    vectors = NumpyVectorStore(dim=2)
    fake = FakeRetriever([_hit("a", 0.9)])
    vectors.upsert(["a"], np.stack([_unit([1, 0])]))
    MmrRetriever(fake, vectors, k=5).retrieve("q")
    assert fake.calls[0][1] == 5


def test_filter_is_forwarded_to_the_wrapped_retriever():
    vectors = NumpyVectorStore(dim=2)
    fake = FakeRetriever([])
    MmrRetriever(fake, vectors).retrieve("the question", filter={"kind": "x"})
    assert fake.calls[0] == ("the question", 10, {"kind": "x"})
