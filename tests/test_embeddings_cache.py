import numpy as np

from nanorag.embeddings.cache import CachingEmbedder, EmbeddingCache, fingerprint
from tests.fakes import FakeEmbedder


class _SpyEmbedder:
    """Wraps a real embedder, recording exactly which texts it was asked to embed."""

    def __init__(self, embedder):
        self._embedder = embedder
        self.dim = embedder.dim
        self.model_id = embedder.model_id
        self.calls: list[list[str]] = []

    def embed(self, texts):
        texts = list(texts)
        self.calls.append(texts)
        return self._embedder.embed(texts)

    def embed_query(self, text):
        return self._embedder.embed_query(text)

    @property
    def call_count(self) -> int:
        return sum(len(c) for c in self.calls)


# --- fingerprint() -----------------------------------------------------


def test_fingerprint_is_deterministic():
    assert fingerprint("hello world") == fingerprint("hello world")


def test_fingerprint_differs_for_different_text():
    assert fingerprint("hello") != fingerprint("world")


def test_fingerprint_is_insensitive_to_line_endings():
    assert fingerprint("a\r\nb") == fingerprint("a\nb")


# --- EmbeddingCache ------------------------------------------------------


def test_get_many_on_empty_cache_returns_nothing(tmp_path):
    cache = EmbeddingCache(tmp_path / "cache.sqlite3")
    assert cache.get_many("model-x", [fingerprint("hello")]) == {}


def test_get_many_of_no_hashes_returns_empty_without_querying(tmp_path):
    cache = EmbeddingCache(tmp_path / "cache.sqlite3")
    assert cache.get_many("model-x", []) == {}


def test_put_many_of_no_items_is_a_no_op(tmp_path):
    cache = EmbeddingCache(tmp_path / "cache.sqlite3")
    cache.put_many("model-x", 3, [])
    assert cache.get_many("model-x", [fingerprint("hello")]) == {}


def test_put_then_get_round_trips_bit_exact(tmp_path):
    cache = EmbeddingCache(tmp_path / "cache.sqlite3")
    vec = np.array([1.0, 2.0, 3.0], dtype=np.float32)
    h = fingerprint("hello")
    cache.put_many("model-x", 3, [(h, vec)])

    found = cache.get_many("model-x", [h])

    assert np.array_equal(found[h], vec)


def test_cache_is_scoped_by_model_id(tmp_path):
    cache = EmbeddingCache(tmp_path / "cache.sqlite3")
    h = fingerprint("hello")
    cache.put_many("model-a", 3, [(h, np.array([1.0, 0.0, 0.0], dtype=np.float32))])

    assert cache.get_many("model-b", [h]) == {}


def test_put_many_upserts_on_conflict(tmp_path):
    cache = EmbeddingCache(tmp_path / "cache.sqlite3")
    h = fingerprint("hello")
    cache.put_many("model-x", 3, [(h, np.array([1.0, 0.0, 0.0], dtype=np.float32))])
    cache.put_many("model-x", 3, [(h, np.array([0.0, 1.0, 0.0], dtype=np.float32))])

    found = cache.get_many("model-x", [h])

    assert np.array_equal(found[h], np.array([0.0, 1.0, 0.0], dtype=np.float32))


def test_cache_persists_across_reopen(tmp_path):
    path = tmp_path / "cache.sqlite3"
    h = fingerprint("hello")
    vec = np.array([1.0, 2.0, 3.0], dtype=np.float32)

    first = EmbeddingCache(path)
    first.put_many("model-x", 3, [(h, vec)])
    first.close()

    reopened = EmbeddingCache(path)
    found = reopened.get_many("model-x", [h])
    assert np.array_equal(found[h], vec)


def test_cache_as_context_manager_closes_on_exit(tmp_path):
    path = tmp_path / "cache.sqlite3"
    with EmbeddingCache(path) as cache:
        cache.put_many(
            "model-x", 3, [(fingerprint("a"), np.zeros(3, dtype=np.float32))]
        )
    # a second connection to the same file must succeed, proving the first closed
    reopened = EmbeddingCache(path)
    assert reopened.get_many("model-x", [fingerprint("a")])


# --- CachingEmbedder -----------------------------------------------------


def test_caching_embedder_returns_same_vectors_as_the_wrapped_embedder(tmp_path):
    inner = FakeEmbedder(dim=8)
    cache = EmbeddingCache(tmp_path / "cache.sqlite3")
    embedder = CachingEmbedder(inner, cache)

    texts = ["alpha", "beta", "gamma"]
    assert np.array_equal(embedder.embed(texts), inner.embed(texts))


def test_second_call_with_the_same_texts_hits_the_cache_entirely(tmp_path):
    spy = _SpyEmbedder(FakeEmbedder(dim=8))
    cache = EmbeddingCache(tmp_path / "cache.sqlite3")
    embedder = CachingEmbedder(spy, cache)
    texts = [f"chunk {i}" for i in range(20)]

    embedder.embed(texts)
    assert spy.call_count == 20

    embedder.embed(texts)
    assert spy.call_count == 20  # no new calls on the second, identical run


def test_cache_hit_survives_a_fresh_embedder_and_cache_instance(tmp_path):
    path = tmp_path / "cache.sqlite3"
    texts = [f"chunk {i}" for i in range(10)]

    spy_1 = _SpyEmbedder(FakeEmbedder(dim=8))
    CachingEmbedder(spy_1, EmbeddingCache(path)).embed(texts)
    assert spy_1.call_count == 10

    spy_2 = _SpyEmbedder(FakeEmbedder(dim=8))
    result = CachingEmbedder(spy_2, EmbeddingCache(path)).embed(texts)
    assert spy_2.call_count == 0
    assert np.array_equal(result, FakeEmbedder(dim=8).embed(texts))


def test_only_the_missing_texts_reach_the_wrapped_embedder(tmp_path):
    spy = _SpyEmbedder(FakeEmbedder(dim=8))
    cache = EmbeddingCache(tmp_path / "cache.sqlite3")
    embedder = CachingEmbedder(spy, cache)

    embedder.embed(["a", "b", "c"])
    assert spy.call_count == 3

    embedder.embed(["b", "c", "d"])
    assert spy.calls[-1] == ["d"]
    assert spy.call_count == 4


def test_a_chunk_with_unchanged_text_hits_cache_even_under_a_different_ordinal(
    tmp_path,
):
    # The whole point of hashing the chunk's own text rather than chunk_id
    # (plan.md §5): the same text served twice, regardless of any id.
    spy = _SpyEmbedder(FakeEmbedder(dim=8))
    cache = EmbeddingCache(tmp_path / "cache.sqlite3")
    embedder = CachingEmbedder(spy, cache)

    embedder.embed(["shared paragraph text"])
    assert spy.call_count == 1
    embedder.embed(["shared paragraph text"])  # same text, would-be different chunk_id
    assert spy.call_count == 1


def test_caching_embedder_of_empty_input_makes_no_calls(tmp_path):
    spy = _SpyEmbedder(FakeEmbedder(dim=8))
    cache = EmbeddingCache(tmp_path / "cache.sqlite3")
    embedder = CachingEmbedder(spy, cache)

    result = embedder.embed([])

    assert result.shape == (0, 8)
    assert spy.calls == []


def test_caching_embedder_never_caches_embed_query(tmp_path):
    spy = _SpyEmbedder(FakeEmbedder(dim=8))
    cache = EmbeddingCache(tmp_path / "cache.sqlite3")
    embedder = CachingEmbedder(spy, cache)

    embedder.embed_query("what is the answer")
    embedder.embed_query("what is the answer")

    assert cache.get_many(spy.model_id, [fingerprint("what is the answer")]) == {}


def test_caching_embedder_exposes_dim_and_model_id(tmp_path):
    inner = FakeEmbedder(dim=8, model_id="fake-x")
    cache = EmbeddingCache(tmp_path / "cache.sqlite3")
    embedder = CachingEmbedder(inner, cache)
    assert embedder.dim == 8
    assert embedder.model_id == "fake-x"
