import numpy as np
import pytest

from nanorag.errors import StoreError
from nanorag.store.numpy_store import NumpyVectorStore


def _unit(vec: list[float]) -> np.ndarray:
    arr = np.array(vec, dtype=np.float32)
    return arr / np.linalg.norm(arr)


def test_construction_rejects_non_positive_dim():
    with pytest.raises(StoreError):
        NumpyVectorStore(dim=0)


def test_dim_property_reflects_construction_argument():
    assert NumpyVectorStore(dim=5).dim == 5


def test_empty_store_has_zero_length_and_empty_search():
    store = NumpyVectorStore(dim=3)
    assert len(store) == 0
    assert store.search(_unit([1, 0, 0]), k=5) == []


def test_upsert_and_search_finds_the_closest_vector():
    store = NumpyVectorStore(dim=3)
    store.upsert(
        ["a", "b", "c"],
        np.stack([_unit([1, 0, 0]), _unit([0, 1, 0]), _unit([0, 0, 1])]),
    )
    results = store.search(_unit([1, 0, 0]), k=1)
    assert results == [("a", pytest.approx(1.0))]


def test_search_orders_results_by_descending_score():
    store = NumpyVectorStore(dim=2)
    store.upsert(
        ["near", "far", "exact"],
        np.stack(
            [_unit([1, 0.1]), _unit([1, 0.5]), _unit([1, 0])],
        ),
    )
    results = store.search(_unit([1, 0]), k=3)
    assert [chunk_id for chunk_id, _ in results] == ["exact", "near", "far"]
    scores = [score for _, score in results]
    assert scores == sorted(scores, reverse=True)


def test_search_respects_k_smaller_than_candidates():
    store = NumpyVectorStore(dim=2)
    store.upsert(
        ["a", "b", "c"], np.stack([_unit([1, 0]), _unit([0, 1]), _unit([1, 1])])
    )
    assert len(store.search(_unit([1, 0]), k=2)) == 2


def test_search_k_larger_than_store_returns_all_candidates():
    store = NumpyVectorStore(dim=2)
    store.upsert(["a", "b"], np.stack([_unit([1, 0]), _unit([0, 1])]))
    assert len(store.search(_unit([1, 0]), k=10)) == 2


def test_upsert_overwrites_existing_id():
    store = NumpyVectorStore(dim=2)
    store.upsert(["a"], np.stack([_unit([1, 0])]))
    store.upsert(["a"], np.stack([_unit([0, 1])]))
    assert len(store) == 1
    results = store.search(_unit([0, 1]), k=1)
    assert results == [("a", pytest.approx(1.0))]


def test_upsert_rejects_wrong_vector_width():
    store = NumpyVectorStore(dim=3)
    with pytest.raises(StoreError):
        store.upsert(["a"], np.stack([_unit([1, 0])]))


def test_upsert_rejects_length_mismatch():
    store = NumpyVectorStore(dim=2)
    with pytest.raises(StoreError):
        store.upsert(["a", "b"], np.stack([_unit([1, 0])]))


def test_upsert_empty_batch_is_a_noop():
    store = NumpyVectorStore(dim=2)
    store.upsert([], np.empty((0, 2), dtype=np.float32))
    assert len(store) == 0


def test_search_rejects_wrong_query_shape():
    store = NumpyVectorStore(dim=3)
    with pytest.raises(StoreError):
        store.search(_unit([1, 0]), k=1)


def test_search_rejects_non_positive_k():
    store = NumpyVectorStore(dim=2)
    store.upsert(["a"], np.stack([_unit([1, 0])]))
    with pytest.raises(StoreError):
        store.search(_unit([1, 0]), k=0)


def test_delete_excludes_from_search_and_reduces_length():
    store = NumpyVectorStore(dim=2)
    store.upsert(["a", "b"], np.stack([_unit([1, 0]), _unit([0, 1])]))
    store.delete(["a"])
    assert len(store) == 1
    results = store.search(_unit([1, 0]), k=5)
    assert [chunk_id for chunk_id, _ in results] == ["b"]


def test_delete_unknown_id_is_a_noop():
    store = NumpyVectorStore(dim=2)
    store.upsert(["a"], np.stack([_unit([1, 0])]))
    store.delete(["does-not-exist"])
    assert len(store) == 1


def test_compact_removes_tombstoned_rows_but_search_is_unchanged():
    store = NumpyVectorStore(dim=2)
    store.upsert(["a", "b"], np.stack([_unit([1, 0]), _unit([0, 1])]))
    store.delete(["a"])
    before = store.search(_unit([1, 0]), k=5)
    store.compact()
    after = store.search(_unit([1, 0]), k=5)
    assert before == after == [("b", pytest.approx(0.0))]
    assert len(store) == 1


def test_compact_is_a_noop_when_nothing_deleted():
    store = NumpyVectorStore(dim=2)
    store.upsert(["a", "b"], np.stack([_unit([1, 0]), _unit([0, 1])]))
    store.compact()
    assert len(store) == 2


def test_upsert_after_compact_reuses_the_reclaimed_id_cleanly():
    store = NumpyVectorStore(dim=2)
    store.upsert(["a", "b"], np.stack([_unit([1, 0]), _unit([0, 1])]))
    store.delete(["a"])
    store.compact()
    store.upsert(["a"], np.stack([_unit([1, 1])]))
    assert len(store) == 2
    results = store.search(_unit([1, 1]), k=1)
    assert results == [("a", pytest.approx(1.0))]


def test_search_allowed_ids_restricts_candidates():
    store = NumpyVectorStore(dim=2)
    store.upsert(
        ["a", "b", "c"], np.stack([_unit([1, 0]), _unit([1, 0]), _unit([1, 0])])
    )
    results = store.search(_unit([1, 0]), k=5, allowed_ids={"b"})
    assert [chunk_id for chunk_id, _ in results] == ["b"]


def test_search_allowed_ids_empty_set_returns_nothing():
    store = NumpyVectorStore(dim=2)
    store.upsert(["a"], np.stack([_unit([1, 0])]))
    assert store.search(_unit([1, 0]), k=5, allowed_ids=set()) == []


def test_search_allowed_ids_survives_compact():
    store = NumpyVectorStore(dim=2)
    store.upsert(
        ["a", "b", "c"], np.stack([_unit([1, 0]), _unit([1, 0]), _unit([1, 0])])
    )
    store.delete(["a"])
    store.compact()
    results = store.search(_unit([1, 0]), k=5, allowed_ids={"b", "does-not-exist"})
    assert [chunk_id for chunk_id, _ in results] == ["b"]


# --- deterministic tie-breaking (Phase C1) ----------------------------------


def test_ties_are_broken_by_chunk_id_not_insertion_order():
    same = _unit([1, 0])
    first = NumpyVectorStore(dim=2)
    first.upsert(["c", "a", "b"], np.stack([same, same, same]))
    second = NumpyVectorStore(dim=2)
    second.upsert(["b", "c", "a"], np.stack([same, same, same]))

    expected = [
        ("a", pytest.approx(1.0)),
        ("b", pytest.approx(1.0)),
        ("c", pytest.approx(1.0)),
    ]
    assert first.search(same, k=3) == expected
    assert second.search(same, k=3) == expected


def test_ties_at_the_k_boundary_are_cut_by_chunk_id():
    # Five identical vectors, k=2: the two smallest ids win, whatever the
    # partition step happened to put at the boundary.
    same = _unit([0, 1])
    store = NumpyVectorStore(dim=2)
    store.upsert(["e", "d", "c", "b", "a"], np.stack([same] * 5))
    assert [cid for cid, _ in store.search(same, k=2)] == ["a", "b"]


def test_score_still_outranks_id_order():
    store = NumpyVectorStore(dim=2)
    store.upsert(["a", "b"], np.stack([_unit([1, 1]), _unit([1, 0])]))
    assert [cid for cid, _ in store.search(_unit([1, 0]), k=2)] == ["b", "a"]


def test_tie_break_survives_compact_and_delete():
    same = _unit([1, 0])
    store = NumpyVectorStore(dim=2)
    store.upsert(["z", "y", "x", "w"], np.stack([same] * 4))
    store.delete(["x"])
    store.compact()
    assert [cid for cid, _ in store.search(same, k=10)] == ["w", "y", "z"]


@pytest.mark.parametrize("allowed_count", [1, 4, 5, 6, 19, 20])
def test_gather_and_in_place_scoring_paths_agree_with_brute_force(allowed_count):
    # search() gathers candidate rows below _GATHER_FRACTION of the index
    # and scores the whole matrix in place above it; both must give exactly
    # the brute-force answer on either side of the threshold (20 rows ->
    # threshold at 5).
    rng = np.random.default_rng(7)
    n, dim = 20, 8
    matrix = rng.standard_normal((n, dim)).astype(np.float32)
    matrix /= np.linalg.norm(matrix, axis=1, keepdims=True)
    ids = [f"id{i:02d}" for i in range(n)]
    store = NumpyVectorStore(dim=dim)
    store.upsert(ids, matrix)
    query = matrix[3]
    allowed = set(ids[:allowed_count])

    scores = matrix @ query
    expected = sorted(
        ((ids[i], float(scores[i])) for i in range(n) if ids[i] in allowed),
        key=lambda r: (-r[1], r[0]),
    )[:3]
    got = store.search(query, k=3, allowed_ids=allowed)
    assert [cid for cid, _ in got] == [cid for cid, _ in expected]
    assert [s for _, s in got] == pytest.approx([s for _, s in expected], abs=1e-6)
