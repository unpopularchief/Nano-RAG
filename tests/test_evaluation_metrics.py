"""``evaluation/retrieval_metrics.py`` and ``answer_metrics.py``: every metric
against hand-computed values on toy rankings — ties (resolved by the store's
chunk-id order, so two equal-score chunks arrive in a fixed order), empty
gold sets (undefined, raised), partial coverage, positions past the list."""

import math

import pytest
from hypothesis import given
from hypothesis import strategies as st

from nanorag.evaluation.answer_metrics import (
    abstained,
    citation_precision,
    citation_validity,
    normalize_answer,
    token_f1,
)
from nanorag.evaluation.retrieval_metrics import (
    coverage_of,
    hit_at_k,
    ndcg_at_k,
    precision_at_k,
    recall_at_k,
    reciprocal_rank,
)
from nanorag.prompting import INSUFFICIENT_CONTEXT_TEXT
from nanorag.types import Answer, Citation, Timings, Usage

R = frozenset  # a relevant position covers gold index/indices
N = frozenset()  # an irrelevant position


# --- ranking metrics, by hand ---------------------------------------------------


def test_metrics_on_a_hand_computed_ranking():
    # gold spans: 0, 1, 2. Ranked list: irrelevant, covers 0, covers 0 and 2,
    # irrelevant, covers 1.
    ranked = [N, R({0}), R({0, 2}), N, R({1})]
    assert recall_at_k(ranked, 3, 1) == 0.0
    assert recall_at_k(ranked, 3, 2) == pytest.approx(1 / 3)
    assert recall_at_k(ranked, 3, 3) == pytest.approx(2 / 3)
    assert recall_at_k(ranked, 3, 5) == 1.0
    assert recall_at_k(ranked, 3, 50) == 1.0
    assert precision_at_k(ranked, 1) == 0.0
    assert precision_at_k(ranked, 3) == pytest.approx(2 / 3)
    assert precision_at_k(ranked, 5) == pytest.approx(3 / 5)
    assert precision_at_k(ranked, 10) == pytest.approx(3 / 10)  # missing = irrelevant
    assert hit_at_k(ranked, 1) == 0.0 and hit_at_k(ranked, 2) == 1.0
    assert reciprocal_rank(ranked) == 0.5
    # nDCG@3: rank 2 covers a new span (0), rank 3 covers 0 again and 2 —
    # new, so it gains too: DCG = 1/log2(3) + 1/log2(4); IDCG = 3 ideal hits.
    dcg = 1 / math.log2(3) + 1 / math.log2(4)
    idcg = 1 / math.log2(2) + 1 / math.log2(3) + 1 / math.log2(4)
    assert ndcg_at_k(ranked, 3, 3) == pytest.approx(dcg / idcg)
    # nDCG@5 with one gold span: ideal list has a single hit at rank 1.
    assert ndcg_at_k([N, R({0})], 1, 5) == pytest.approx((1 / math.log2(3)) / 1.0)


def test_ndcg_gains_only_for_newly_covered_spans():
    # Overlapping chunks: two positions cover the same single gold span.
    assert ndcg_at_k([R({0}), R({0})], 1, 2) == 1.0
    # ...and a repeat before the second span's first coverage costs rank.
    twice_then_new = [R({0}), R({0}), R({1})]
    expected = (1 / math.log2(2) + 1 / math.log2(4)) / (
        1 / math.log2(2) + 1 / math.log2(3)
    )
    assert ndcg_at_k(twice_then_new, 2, 3) == pytest.approx(expected)


def test_perfect_and_empty_rankings():
    perfect = [R({0}), R({1})]
    assert recall_at_k(perfect, 2, 2) == 1.0
    assert precision_at_k(perfect, 2) == 1.0
    assert ndcg_at_k(perfect, 2, 2) == 1.0
    assert reciprocal_rank(perfect) == 1.0
    nothing = [N, N, N]
    assert recall_at_k(nothing, 2, 3) == 0.0
    assert precision_at_k(nothing, 3) == 0.0
    assert hit_at_k(nothing, 3) == 0.0
    assert reciprocal_rank(nothing) == 0.0
    assert ndcg_at_k(nothing, 2, 3) == 0.0
    assert reciprocal_rank([]) == 0.0 and recall_at_k([], 1, 5) == 0.0


def test_ties_are_graded_in_the_order_the_store_resolved_them():
    # Two chunks with the same score: the retriever breaks the tie by chunk
    # id, so the metric sees a definite order and grades it as given.
    tied_first = [R({0}), N]
    tied_second = [N, R({0})]
    assert reciprocal_rank(tied_first) == 1.0
    assert reciprocal_rank(tied_second) == 0.5
    assert ndcg_at_k(tied_first, 1, 2) == 1.0
    assert ndcg_at_k(tied_second, 1, 2) == pytest.approx(1 / math.log2(3))


def test_empty_gold_set_is_undefined_not_zero():
    for call in (
        lambda: recall_at_k([N], 0, 1),
        lambda: ndcg_at_k([N], 0, 1),
    ):
        with pytest.raises(ValueError, match="no gold"):
            call()
    with pytest.raises(ValueError, match="k must be"):
        precision_at_k([N], 0)
    with pytest.raises(ValueError, match="out of range"):
        recall_at_k([R({3})], 2, 1)


@given(
    n_gold=st.integers(min_value=1, max_value=4),
    ranked=st.lists(
        st.frozensets(st.integers(min_value=0, max_value=3), max_size=4), max_size=8
    ),
    k=st.integers(min_value=1, max_value=10),
)
def test_metric_invariants(n_gold, ranked, k):
    ranked = [frozenset(i for i in c if i < n_gold) for c in ranked]
    recall = recall_at_k(ranked, n_gold, k)
    assert 0.0 <= recall <= 1.0
    assert recall <= recall_at_k(ranked, n_gold, k + 1)  # monotone in k
    assert 0.0 <= precision_at_k(ranked, k) <= 1.0
    assert hit_at_k(ranked, k) == (1.0 if any(ranked[:k]) else 0.0)
    assert 0.0 <= ndcg_at_k(ranked, n_gold, k) <= 1.0
    rr = reciprocal_rank(ranked)
    assert rr == 0.0 or 1.0 / rr == int(1.0 / rr)


def test_coverage_of_is_half_open_and_document_scoped():
    gold = [("doc-a", 10, 20), ("doc-a", 30, 40), ("doc-b", 10, 20)]
    assert coverage_of("doc-a", 0, 10, gold) == N  # ends where gold starts
    assert coverage_of("doc-a", 0, 11, gold) == R({0})
    assert coverage_of("doc-a", 19, 31, gold) == R({0, 1})
    assert coverage_of("doc-a", 20, 30, gold) == N  # the gap between spans
    assert coverage_of("doc-b", 15, 16, gold) == R({2})
    assert coverage_of("doc-c", 0, 100, gold) == N


# --- answer metrics -------------------------------------------------------------


def _answer(text, citations=()):
    return Answer(
        text=text,
        citations=tuple(citations),
        contexts=(),
        insufficient_context=False,
        truncated=False,
        usage=Usage("fake", "m", 1, 1, 2),
        timings=Timings(),
    )


def _cite(label, doc="d", start=0, end=10):
    return Citation(label, f"c{label}", doc, f"{doc}.md", start, end)


def test_citation_validity_counts_markers_of_both_bracket_styles():
    assert citation_validity(_answer("no markers")) is None
    answer = _answer("a [1] b 【2】 c [9] d [1]", [_cite(1), _cite(2)])
    assert citation_validity(answer) == pytest.approx(3 / 4)


def test_citation_precision_against_gold_spans():
    gold = [("d", 5, 15)]
    assert citation_precision(_answer("x"), gold) is None
    on = _cite(1, "d", 0, 10)
    off = _cite(2, "d", 20, 30)
    other_doc = _cite(3, "e", 0, 10)
    assert citation_precision(_answer("x", [on]), gold) == 1.0
    mixed = _answer("x", [on, off, other_doc])
    assert citation_precision(mixed, gold) == pytest.approx(1 / 3)


def test_abstained_matches_the_fixed_text_modulo_whitespace():
    assert abstained(_answer(INSUFFICIENT_CONTEXT_TEXT))
    wrapped = "  " + INSUFFICIENT_CONTEXT_TEXT.replace(" ", "\n") + "\n"
    assert abstained(_answer(wrapped))
    assert not abstained(_answer("I don't know, sorry."))


def test_token_f1_squad_normalisation():
    assert normalize_answer("The Quick, brown fox!") == ["quick", "brown", "fox"]
    assert token_f1("the quick brown fox", "a quick brown fox") == 1.0
    assert token_f1("quick fox", "quick brown fox jumps") == pytest.approx(
        2 * (1.0 * 0.5) / (1.0 + 0.5)
    )
    assert token_f1("nothing in common", "quick fox") == 0.0
    assert token_f1("", "") == 1.0 and token_f1("", "x") == 0.0
