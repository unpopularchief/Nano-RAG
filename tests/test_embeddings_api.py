"""Hosted embedding clients exercise their wire formats without network calls."""

import json

import httpx
import numpy as np
import pytest

from nanorag.embeddings import GeminiEmbedder, JinaEmbedder, OpenAICompatEmbedder
from nanorag.errors import AuthError, ConfigError, ProviderError, RateLimitError
from nanorag.ratelimit import Backoff


class Recorder:
    """Record HTTP requests and return a scripted response."""

    def __init__(self, status: int, body: dict, headers: dict | None = None) -> None:
        self.status = status
        self.body = body
        self.headers = headers or {}
        self.requests: list[httpx.Request] = []

    def __call__(self, request: httpx.Request) -> httpx.Response:
        self.requests.append(request)
        return httpx.Response(self.status, json=self.body, headers=self.headers)


def _transport(recorder: Recorder) -> httpx.MockTransport:
    return httpx.MockTransport(recorder)


def _gemini(rows: list[list[float]], recorder: Recorder) -> GeminiEmbedder:
    recorder.body = {"embeddings": [{"values": row} for row in rows]}
    return GeminiEmbedder(api_key="secret", dim=2, transport=_transport(recorder))


def _jina(rows: list[list[float]], recorder: Recorder) -> JinaEmbedder:
    recorder.body = {
        "data": [{"index": i, "embedding": row} for i, row in enumerate(rows)]
    }
    return JinaEmbedder(api_key="secret", dim=2, transport=_transport(recorder))


def _openai(rows: list[list[float]], recorder: Recorder) -> OpenAICompatEmbedder:
    recorder.body = {
        "data": [{"index": i, "embedding": row} for i, row in enumerate(rows)]
    }
    return OpenAICompatEmbedder(api_key="secret", dim=2, transport=_transport(recorder))


@pytest.mark.parametrize(
    "factory,endpoint,expected_url",
    [
        (
            _gemini,
            "gemini",
            "https://generativelanguage.googleapis.com/v1beta/models/gemini-embedding-2:batchEmbedContents",
        ),
        (_jina, "jina", "https://api.jina.ai/v1/embeddings"),
        (_openai, "openai", "https://api.openai.com/v1/embeddings"),
    ],
)
def test_hosted_embedders_return_normalized_float32_and_send_batches(
    factory, endpoint, expected_url
):
    recorder = Recorder(200, {})
    embedder = factory([[3, 4], [0, 2]], recorder)

    vectors = embedder.embed(["first", "second"])

    assert isinstance(embedder, (GeminiEmbedder, JinaEmbedder, OpenAICompatEmbedder))
    assert vectors.dtype == np.float32
    assert vectors.shape == (2, 2)
    np.testing.assert_allclose(np.linalg.norm(vectors, axis=1), [1, 1])
    request = recorder.requests[0]
    assert str(request.url) == expected_url
    assert request.headers[
        "Authorization" if endpoint != "gemini" else "x-goog-api-key"
    ]
    assert (
        len(
            json.loads(request.content)["input" if endpoint != "gemini" else "requests"]
        )
        == 2
    )


def test_provider_specific_query_and_document_tasks_and_stable_model_ids():
    rec_g = Recorder(200, {"embeddings": [{"values": [3, 4]}]})
    gemini = GeminiEmbedder(api_key="k", dim=2, transport=_transport(rec_g))
    assert gemini.embed_query("q").shape == (2,)
    gemini.embed(["d"])
    g_bodies = [json.loads(req.content) for req in rec_g.requests]
    assert (
        g_bodies[0]["requests"][0]["embedContentConfig"]["taskType"]
        == "RETRIEVAL_QUERY"
    )
    assert (
        g_bodies[1]["requests"][0]["embedContentConfig"]["taskType"]
        == "RETRIEVAL_DOCUMENT"
    )
    assert gemini.model_id == "gemini:gemini-embedding-2:2"

    rec_j = Recorder(200, {"data": [{"index": 0, "embedding": [3, 4]}]})
    jina = JinaEmbedder(api_key="k", dim=2, transport=_transport(rec_j))
    jina.embed_query("q")
    jina.embed(["d"])
    assert [json.loads(r.content)["task"] for r in rec_j.requests] == [
        "retrieval.query",
        "retrieval.passage",
    ]
    assert jina.model_id == "jina:jina-embeddings-v3:2"


def test_provider_indexes_restore_input_order():
    body = {
        "data": [{"index": 1, "embedding": [0, 2]}, {"index": 0, "embedding": [3, 4]}]
    }
    embedder = OpenAICompatEmbedder(
        api_key="k",
        dim=2,
        transport=httpx.MockTransport(lambda r: httpx.Response(200, json=body)),
    )
    result = embedder.embed(["a", "b"])
    np.testing.assert_allclose(result, [[0.6, 0.8], [0, 1]])


def test_empty_batches_do_not_make_requests_and_custom_openai_endpoint_is_supported():
    recorder = Recorder(200, {"data": []})
    embedder = OpenAICompatEmbedder(
        api_key="k",
        api_key_env="CUSTOM_KEY",
        base_url="https://vectors.example/v1/",
        model="my-model",
        dim=2,
        transport=_transport(recorder),
    )
    assert embedder.embed([]).shape == (0, 2)
    assert recorder.requests == []
    assert embedder.model_id == "https://vectors.example/v1:my-model:2"


def test_key_is_required_and_never_appears_in_repr_or_provider_errors(monkeypatch):
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    with pytest.raises(ConfigError, match="GEMINI_API_KEY"):
        GeminiEmbedder()
    recorder = Recorder(401, {"error": {"message": "bad credentials"}})
    embedder = OpenAICompatEmbedder(
        api_key="secret", dim=2, transport=_transport(recorder), backoff=Backoff(0)
    )
    assert "secret" not in repr(embedder)
    with pytest.raises(AuthError) as error:
        embedder.embed(["text"])
    assert "secret" not in str(error.value)
    assert "secret" not in repr(error.value)


@pytest.mark.parametrize(
    "status,expected",
    [(429, RateLimitError), (503, ProviderError), (400, ProviderError)],
)
def test_http_failures_are_mapped_to_typed_provider_errors(status, expected):
    recorder = Recorder(status, {"error": "failed"}, {"Retry-After": "2"})
    embedder = JinaEmbedder(
        api_key="k", dim=2, transport=_transport(recorder), backoff=Backoff(0)
    )
    with pytest.raises(expected) as error:
        embedder.embed(["text"])
    if status == 429:
        assert error.value.retry_after == 2


@pytest.mark.parametrize(
    "body",
    [
        {"data": [{"index": 0, "embedding": [1]}]},
        {"data": [{"index": 0, "embedding": ["nan", 1]}]},
        {"data": []},
        {"wrong": []},
    ],
)
def test_invalid_provider_vectors_raise_provider_error(body):
    embedder = OpenAICompatEmbedder(
        api_key="k",
        dim=2,
        transport=httpx.MockTransport(lambda r: httpx.Response(200, json=body)),
    )
    with pytest.raises(ProviderError):
        embedder.embed(["text"])


def test_importing_embedding_package_does_not_import_local_embedding_runtime():
    import sys

    import nanorag.embeddings  # noqa: F401

    assert "fastembed" not in sys.modules
    assert "onnxruntime" not in sys.modules
