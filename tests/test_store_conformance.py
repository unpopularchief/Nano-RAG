"""Shared ``VectorStore`` conformance suite (plan.md §9 Phase G session G2).

Every test here runs, unmodified, against all three backends via the
``store_factory`` fixture: the in-memory ``NumpyVectorStore`` and the two
external adapters. This is what "swapping the store changes one constructor
line" actually means — same behaviour, different backend.

``-m integration`` (needs Docker services for the two external backends;
see ``.github/workflows/integration.yml``), deselected by default. Only
``nanorag.store.external.*`` wrapper modules are imported at module scope —
never ``qdrant_client``/``pg8000`` directly — so collecting this file under
the default, deselecting run still costs nothing (mirrors
``tests/test_rerank_local_cross_encoder.py``'s pattern for ``-m local``).

Per plan.md §18 F7, this suite checks **behaviour parity, not recall
parity** — Qdrant is ANN (HNSW) by default, though its ``full_scan_threshold``
means every fixture here (a handful of vectors) is searched exactly under
the hood anyway. Tie-break order across candidates with an identical score
is *not* asserted here (only ``NumpyVectorStore``'s own test suite asserts
its documented ascending-chunk-id tie-break) — every fixture below is built
so no two candidates ever tie.
"""

from __future__ import annotations

import os
import uuid
from collections.abc import Callable

import numpy as np
import pytest

from nanorag.errors import StoreError
from nanorag.store.base import VectorStore
from nanorag.store.numpy_store import NumpyVectorStore

pytestmark = pytest.mark.integration

#: Tolerance for score comparisons: external backends compute the dot
#: product through their own numeric path, not bit-identical to NumPy's.
_TOL = 1e-4


def _unit(vec: list[float]) -> np.ndarray:
    arr = np.array(vec, dtype=np.float32)
    return arr / np.linalg.norm(arr)


def _make_qdrant(dim: int) -> VectorStore:
    from nanorag.store.external.qdrant import QdrantVectorStore

    url = os.environ.get("QDRANT_URL", "http://localhost:6333")
    return QdrantVectorStore(
        dim=dim, collection_name=f"nanorag-test-{uuid.uuid4().hex}", url=url
    )


def _make_pgvector(dim: int) -> VectorStore:
    from nanorag.store.external.pgvector import PgVectorStore

    dsn = os.environ.get(
        "PGVECTOR_DSN", "postgresql://nanorag:nanorag@localhost:5432/nanorag"
    )
    return PgVectorStore(
        dim=dim, dsn=dsn, table_name=f"nanorag_test_{uuid.uuid4().hex}"
    )


_FACTORIES: dict[str, Callable[[int], VectorStore]] = {
    "numpy": lambda dim: NumpyVectorStore(dim=dim),
    "qdrant": _make_qdrant,
    "pgvector": _make_pgvector,
}


@pytest.fixture(params=["numpy", "qdrant", "pgvector"])
def store_factory(request: pytest.FixtureRequest) -> Callable[[int], VectorStore]:
    """Return a ``dim -> VectorStore`` factory for one backend."""
    return _FACTORIES[request.param]


def test_dim_property_reflects_construction_argument(store_factory):
    assert store_factory(5).dim == 5


def test_empty_store_search_returns_nothing(store_factory):
    store = store_factory(3)
    assert store.search(_unit([1, 0, 0]), k=5) == []


def test_upsert_and_search_finds_the_closest_vector(store_factory):
    store = store_factory(3)
    store.upsert(
        ["a", "b", "c"],
        np.stack([_unit([1, 0, 0]), _unit([0, 1, 0]), _unit([0, 0, 1])]),
    )
    results = store.search(_unit([1, 0, 0]), k=1)
    assert results == [("a", pytest.approx(1.0, abs=_TOL))]


def test_search_orders_results_by_descending_score(store_factory):
    store = store_factory(2)
    store.upsert(
        ["near", "far", "exact"],
        np.stack([_unit([1, 0.1]), _unit([1, 0.5]), _unit([1, 0])]),
    )
    results = store.search(_unit([1, 0]), k=3)
    assert [chunk_id for chunk_id, _ in results] == ["exact", "near", "far"]
    scores = [score for _, score in results]
    assert scores == sorted(scores, reverse=True)


def test_search_respects_k_smaller_than_candidates(store_factory):
    store = store_factory(2)
    store.upsert(
        ["a", "b", "c"], np.stack([_unit([1, 0]), _unit([0, 1]), _unit([1, 2])])
    )
    assert len(store.search(_unit([1, 0]), k=2)) == 2


def test_search_k_larger_than_store_returns_all_candidates(store_factory):
    store = store_factory(2)
    store.upsert(["a", "b"], np.stack([_unit([1, 0]), _unit([0, 1])]))
    assert len(store.search(_unit([1, 0]), k=10)) == 2


def test_upsert_overwrites_existing_id(store_factory):
    store = store_factory(2)
    store.upsert(["a"], np.stack([_unit([1, 0])]))
    store.upsert(["a"], np.stack([_unit([0, 1])]))
    results = store.search(_unit([0, 1]), k=5)
    assert results == [("a", pytest.approx(1.0, abs=_TOL))]


def test_upsert_rejects_wrong_vector_width(store_factory):
    store = store_factory(3)
    with pytest.raises(StoreError):
        store.upsert(["a"], np.stack([_unit([1, 0])]))


def test_upsert_rejects_length_mismatch(store_factory):
    store = store_factory(2)
    with pytest.raises(StoreError):
        store.upsert(["a", "b"], np.stack([_unit([1, 0])]))


def test_upsert_empty_batch_is_a_noop(store_factory):
    store = store_factory(2)
    store.upsert([], np.empty((0, 2), dtype=np.float32))
    assert store.search(_unit([1, 0]), k=5) == []


def test_search_rejects_wrong_query_shape(store_factory):
    store = store_factory(3)
    with pytest.raises(StoreError):
        store.search(_unit([1, 0]), k=1)


def test_search_rejects_non_positive_k(store_factory):
    store = store_factory(2)
    store.upsert(["a"], np.stack([_unit([1, 0])]))
    with pytest.raises(StoreError):
        store.search(_unit([1, 0]), k=0)


def test_delete_excludes_from_search(store_factory):
    store = store_factory(2)
    store.upsert(["a", "b"], np.stack([_unit([1, 0]), _unit([0, 1])]))
    store.delete(["a"])
    results = store.search(_unit([1, 0]), k=5)
    assert [chunk_id for chunk_id, _ in results] == ["b"]


def test_delete_unknown_id_is_a_noop(store_factory):
    store = store_factory(2)
    store.upsert(["a"], np.stack([_unit([1, 0])]))
    store.delete(["does-not-exist"])
    results = store.search(_unit([1, 0]), k=5)
    assert [chunk_id for chunk_id, _ in results] == ["a"]


def test_compact_does_not_change_search_results(store_factory):
    store = store_factory(2)
    store.upsert(["a", "b"], np.stack([_unit([1, 0]), _unit([0, 1])]))
    store.delete(["a"])
    before = store.search(_unit([1, 0]), k=5)
    store.compact()
    after = store.search(_unit([1, 0]), k=5)
    assert before == after == [("b", pytest.approx(0.0, abs=_TOL))]


def test_compact_is_safe_when_nothing_deleted(store_factory):
    store = store_factory(2)
    store.upsert(["a", "b"], np.stack([_unit([1, 0]), _unit([0, 1])]))
    before = store.search(_unit([1, 0]), k=5)
    store.compact()
    after = store.search(_unit([1, 0]), k=5)
    assert before == after


def test_search_allowed_ids_restricts_candidates(store_factory):
    store = store_factory(2)
    store.upsert(
        ["a", "b", "c"], np.stack([_unit([1, 0]), _unit([1, 0]), _unit([1, 0])])
    )
    results = store.search(_unit([1, 0]), k=5, allowed_ids={"b"})
    assert [chunk_id for chunk_id, _ in results] == ["b"]


def test_search_allowed_ids_empty_set_returns_nothing(store_factory):
    store = store_factory(2)
    store.upsert(["a"], np.stack([_unit([1, 0])]))
    assert store.search(_unit([1, 0]), k=5, allowed_ids=set()) == []


def test_get_vectors_returns_stored_vectors(store_factory):
    store = store_factory(2)
    a, b = _unit([1, 0]), _unit([0, 1])
    store.upsert(["a", "b"], np.stack([a, b]))
    got = store.get_vectors(["a", "b"])
    assert set(got) == {"a", "b"}
    np.testing.assert_allclose(got["a"], a, atol=_TOL)
    np.testing.assert_allclose(got["b"], b, atol=_TOL)


def test_get_vectors_omits_unknown_and_deleted_ids(store_factory):
    store = store_factory(2)
    store.upsert(["a", "b"], np.stack([_unit([1, 0]), _unit([0, 1])]))
    store.delete(["b"])
    got = store.get_vectors(["a", "b", "nope"])
    assert set(got) == {"a"}


def test_get_vectors_empty_request_or_empty_store(store_factory):
    store = store_factory(2)
    assert store.get_vectors([]) == {}
    store.upsert(["a"], np.stack([_unit([1, 0])]))
    assert store.get_vectors([]) == {}
