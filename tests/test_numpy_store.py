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
