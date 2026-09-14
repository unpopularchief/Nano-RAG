"""Real-ONNX-model tests for ``FastEmbedEmbedder`` (plan.md §9 Phase B, B4).

Opt-in (``-m local``): offline after the first download, but slow enough
(and network-dependent on a cold cache) that it does not belong in the
default suite (plan.md §12). Run explicitly with::

    uv run pytest -m local tests/test_embeddings_local.py

The 10k-chunk checkpoint test uses ``BAAI/bge-small-en-v1.5`` rather than the
documented default ``BAAI/bge-base-en-v1.5`` — same code path, same ONNX
Runtime, a quarter the compute, so the opt-in suite stays fast. A separate,
much smaller test exercises the actual default model id end-to-end.
"""

from __future__ import annotations

import numpy as np
import pytest

from nanorag.embeddings.batching import BatchingEmbedder
from nanorag.embeddings.cache import CachingEmbedder, EmbeddingCache
from nanorag.embeddings.local import DEFAULT_MODEL_ID, FastEmbedEmbedder
from nanorag.errors import ConfigError, EmbeddingError

pytestmark = pytest.mark.local

_SMALL_MODEL = "BAAI/bge-small-en-v1.5"
_SMALL_DIM = 384


def test_construction_resolves_dim_without_a_full_download_first():
    embedder = FastEmbedEmbedder(_SMALL_MODEL)
    assert embedder.dim == _SMALL_DIM
    assert embedder.model_id == _SMALL_MODEL


def test_unknown_model_id_raises_config_error():
    with pytest.raises(ConfigError):
        FastEmbedEmbedder("not/a-real-model")


def test_embed_returns_l2_normalised_float32_vectors():
    embedder = FastEmbedEmbedder(_SMALL_MODEL)
    vectors = embedder.embed(["hello world", "a second sentence"])
    assert vectors.shape == (2, _SMALL_DIM)
    assert vectors.dtype == np.float32
    norms = np.linalg.norm(vectors, axis=1)
    assert np.allclose(norms, 1.0, atol=1e-4)


def test_embed_of_empty_sequence_returns_empty_array():
    embedder = FastEmbedEmbedder(_SMALL_MODEL)
    result = embedder.embed([])
    assert result.shape == (0, _SMALL_DIM)


def test_embed_of_empty_string_does_not_crash():
    embedder = FastEmbedEmbedder(_SMALL_MODEL)
    vectors = embedder.embed([""])
    assert vectors.shape == (1, _SMALL_DIM)
    assert np.all(np.isfinite(vectors))


def test_embed_query_returns_a_unit_vector():
    embedder = FastEmbedEmbedder(_SMALL_MODEL)
    vector = embedder.embed_query("what is nano rag?")
    assert vector.shape == (_SMALL_DIM,)
    assert np.isclose(float(np.linalg.norm(vector)), 1.0, atol=1e-4)


def test_two_runs_of_the_same_text_are_deterministic_on_this_build():
    # plan.md §18 F5: deterministic on a pinned build, not bit-identical
    # across builds — both calls here are the same process/build, so they
    # must match exactly.
    embedder = FastEmbedEmbedder(_SMALL_MODEL)
    first = embedder.embed(["determinism check"])
    second = embedder.embed(["determinism check"])
    assert np.array_equal(first, second)


def test_default_model_id_matches_the_documented_plan_default():
    assert DEFAULT_MODEL_ID == "BAAI/bge-base-en-v1.5"


def test_the_actual_default_model_constructs_and_embeds():
    embedder = FastEmbedEmbedder()  # BAAI/bge-base-en-v1.5
    assert embedder.dim == 768
    vectors = embedder.embed(["a small sanity check for the documented default"])
    assert vectors.shape == (1, 768)
    assert np.isclose(float(np.linalg.norm(vectors[0])), 1.0, atol=1e-4)


# --- the B4 checkpoint: "10k chunks embedded locally; second run hits cache
# --- 100%" ------------------------------------------------------------


def test_10k_chunks_embed_locally_and_a_second_run_hits_cache_entirely(tmp_path):
    texts = [f"chunk number {i} discusses topic {i % 47}." for i in range(10_000)]
    cache_path = tmp_path / "embedding_cache.sqlite3"

    class _CountingWrapper:
        """Counts every text handed to the real embedder underneath."""

        def __init__(self, inner):
            self._inner = inner
            self.dim = inner.dim
            self.model_id = inner.model_id
            self.embedded_count = 0

        def embed(self, batch):
            self.embedded_count += len(batch)
            return self._inner.embed(batch)

        def embed_query(self, text):
            return self._inner.embed_query(text)

    real = FastEmbedEmbedder(_SMALL_MODEL)

    first_wrapper = _CountingWrapper(real)
    first_run = CachingEmbedder(
        BatchingEmbedder(first_wrapper, batch_size=256), EmbeddingCache(cache_path)
    )
    first_vectors = first_run.embed(texts)
    assert first_wrapper.embedded_count == 10_000
    assert first_vectors.shape == (10_000, _SMALL_DIM)

    second_wrapper = _CountingWrapper(real)
    second_run = CachingEmbedder(
        BatchingEmbedder(second_wrapper, batch_size=256), EmbeddingCache(cache_path)
    )
    second_vectors = second_run.embed(texts)

    assert second_wrapper.embedded_count == 0  # 100% cache hit, no model calls
    assert np.array_equal(first_vectors, second_vectors)


def test_a_broken_underlying_model_raises_embedding_error(monkeypatch):
    embedder = FastEmbedEmbedder(_SMALL_MODEL)

    def _boom(*args, **kwargs):
        raise RuntimeError("onnxruntime blew up")

    monkeypatch.setattr(embedder._model, "embed", _boom)
    with pytest.raises(EmbeddingError):
        embedder.embed(["this will fail"])


def test_a_broken_underlying_model_raises_embedding_error_for_queries(monkeypatch):
    embedder = FastEmbedEmbedder(_SMALL_MODEL)

    def _boom(*args, **kwargs):
        raise RuntimeError("onnxruntime blew up")

    monkeypatch.setattr(embedder._model, "query_embed", _boom)
    with pytest.raises(EmbeddingError):
        embedder.embed_query("this will fail")
