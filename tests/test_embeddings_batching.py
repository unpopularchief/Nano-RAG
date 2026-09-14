import numpy as np
import pytest

from nanorag.embeddings.batching import BatchingEmbedder, batched
from tests.fakes import FakeEmbedder


class _SpyEmbedder:
    """Wraps a real embedder, recording the size of every ``embed`` call."""

    def __init__(self, embedder):
        self._embedder = embedder
        self.dim = embedder.dim
        self.model_id = embedder.model_id
        self.batch_sizes: list[int] = []

    def embed(self, texts):
        self.batch_sizes.append(len(texts))
        return self._embedder.embed(texts)

    def embed_query(self, text):
        return self._embedder.embed_query(text)


# --- batched() -------------------------------------------------------------


def test_batched_splits_into_slices_of_at_most_batch_size():
    items = [str(i) for i in range(10)]
    result = list(batched(items, 3))
    assert result == [
        ["0", "1", "2"],
        ["3", "4", "5"],
        ["6", "7", "8"],
        ["9"],
    ]


def test_batched_of_empty_sequence_yields_nothing():
    assert list(batched([], 3)) == []


def test_batched_batch_size_larger_than_input_yields_one_batch():
    assert list(batched(["a", "b"], 10)) == [["a", "b"]]


def test_batched_rejects_non_positive_batch_size():
    with pytest.raises(ValueError):
        list(batched(["a"], 0))


# --- BatchingEmbedder --------------------------------------------------


def test_batching_embedder_calls_the_wrapped_embedder_in_bounded_batches():
    spy = _SpyEmbedder(FakeEmbedder(dim=8))
    embedder = BatchingEmbedder(spy, batch_size=4)
    texts = [f"chunk {i}" for i in range(10)]

    embedder.embed(texts)

    assert spy.batch_sizes == [4, 4, 2]


def test_batching_embedder_result_matches_one_unbatched_call():
    inner = FakeEmbedder(dim=8)
    texts = [f"chunk {i}" for i in range(10)]

    batched_result = BatchingEmbedder(inner, batch_size=3).embed(texts)
    direct_result = inner.embed(texts)

    assert np.array_equal(batched_result, direct_result)


def test_batching_embedder_exposes_dim_and_model_id():
    inner = FakeEmbedder(dim=8, model_id="fake-x")
    embedder = BatchingEmbedder(inner, batch_size=4)
    assert embedder.dim == 8
    assert embedder.model_id == "fake-x"


def test_batching_embedder_of_empty_input_returns_empty_and_makes_no_calls():
    spy = _SpyEmbedder(FakeEmbedder(dim=8))
    embedder = BatchingEmbedder(spy, batch_size=4)
    result = embedder.embed([])
    assert result.shape == (0, 8)
    assert spy.batch_sizes == []


def test_batching_embedder_delegates_embed_query_unbatched():
    inner = FakeEmbedder(dim=8)
    embedder = BatchingEmbedder(inner, batch_size=4)
    assert np.array_equal(embedder.embed_query("hello"), inner.embed_query("hello"))


def test_batching_embedder_rejects_non_positive_batch_size():
    with pytest.raises(ValueError):
        BatchingEmbedder(FakeEmbedder(), batch_size=0)
