"""``JinaReranker`` against recorded responses via ``MockTransport``: request
shape, result remapping, the error mapping (401/429/5xx/malformed body, all
translated to ``RerankError`` at the ``rerank()`` boundary), retry
behaviour, and that the key never leaks. See the module docstring in
``nanorag/rerank/jina.py`` for why this mapping is inferred rather than
observed against a live key (no ``JINA_API_KEY`` was available this
session)."""

import json

import httpx
import pytest

from nanorag.errors import ConfigError, RerankError
from nanorag.hashing import content_hash, stable_doc_id
from nanorag.ratelimit import Backoff
from nanorag.rerank.base import Reranker
from nanorag.rerank.jina import DEFAULT_MODEL, JinaReranker
from nanorag.types import Chunk, Document, ScoredChunk

_DOC = Document(
    doc_id=stable_doc_id("d.txt"),
    source_uri="d.txt",
    text="text",
    content_hash=content_hash("text"),
    metadata={},
)


def _hit(ordinal: int, text: str, score: float = 0.5) -> ScoredChunk:
    return ScoredChunk(
        chunk=Chunk(
            chunk_id=f"c{ordinal}",
            doc_id=_DOC.doc_id,
            ordinal=ordinal,
            text=text,
            start_char=0,
            end_char=len(text),
            token_count=1,
        ),
        score=score,
        source="dense",
    )


class Recorder:
    """A MockTransport handler that returns scripted responses and records requests."""

    def __init__(self, *responses):
        self.responses = list(responses)
        self.requests: list[httpx.Request] = []

    def __call__(self, request: httpx.Request) -> httpx.Response:
        self.requests.append(request)
        item = self.responses.pop(0) if len(self.responses) > 1 else self.responses[0]
        if isinstance(item, Exception):
            raise item
        status, body, headers = item
        return httpx.Response(status, json=body, headers=headers or {})


def _reranker(*responses, **kwargs):
    recorder = Recorder(*responses)
    kwargs.setdefault("api_key", "jina-test-secret")
    kwargs.setdefault("backoff", Backoff(max_retries=2, base_delay=0.01))
    kwargs.setdefault("sleep", lambda s: None)
    reranker = JinaReranker(transport=httpx.MockTransport(recorder), **kwargs)
    return reranker, recorder


def _ok(results):
    return {"model": DEFAULT_MODEL, "usage": {"total_tokens": 12}, "results": results}


# --- protocol / construction ----------------------------------------------------


def test_jina_reranker_satisfies_the_protocol():
    reranker, _ = _reranker((200, _ok([]), None))
    assert isinstance(reranker, Reranker)


def test_key_comes_from_the_environment(monkeypatch):
    monkeypatch.delenv("JINA_API_KEY", raising=False)
    with pytest.raises(ConfigError, match="JINA_API_KEY"):
        JinaReranker(transport=httpx.MockTransport(Recorder()))
    monkeypatch.setenv("JINA_API_KEY", "from-env")
    reranker = JinaReranker(transport=httpx.MockTransport(Recorder()))
    assert reranker._client.headers["Authorization"] == "Bearer from-env"


def test_repr_and_errors_never_contain_the_key():
    reranker, _ = _reranker((401, {"detail": "bad key"}, None))
    assert "jina-test-secret" not in repr(reranker)
    with pytest.raises(RerankError) as info:
        reranker.rerank("q", [_hit(0, "a")], top_n=1)
    assert "jina-test-secret" not in str(info.value)


# --- the happy path -----------------------------------------------------------


def test_sends_query_documents_and_top_n_and_remaps_results_by_index():
    hits = [_hit(0, "alpha"), _hit(1, "beta"), _hit(2, "gamma")]
    reranker, rec = _reranker(
        (
            200,
            _ok(
                [
                    {"index": 2, "relevance_score": 0.95},
                    {"index": 0, "relevance_score": 0.40},
                ]
            ),
            None,
        )
    )

    out = reranker.rerank("what is gamma?", hits, top_n=2)

    body = json.loads(rec.requests[0].content)
    assert body == {
        "model": DEFAULT_MODEL,
        "query": "what is gamma?",
        "top_n": 2,
        "documents": ["alpha", "beta", "gamma"],
    }
    assert [h.chunk.text for h in out] == ["gamma", "alpha"]
    assert [h.score for h in out] == [0.95, 0.40]
    assert all(h.source == "rerank:jina" for h in out)


def test_results_are_sorted_score_descending_even_if_the_api_sends_them_unsorted():
    hits = [_hit(0, "a"), _hit(1, "b")]
    reranker, _ = _reranker(
        (
            200,
            _ok(
                [
                    {"index": 0, "relevance_score": 0.1},
                    {"index": 1, "relevance_score": 0.9},
                ]
            ),
            None,
        )
    )
    out = reranker.rerank("q", hits, top_n=2)
    assert [h.chunk.text for h in out] == ["b", "a"]


def test_top_n_slices_even_if_the_api_returns_more_results():
    hits = [_hit(0, "a"), _hit(1, "b")]
    reranker, _ = _reranker(
        (
            200,
            _ok(
                [
                    {"index": 0, "relevance_score": 0.9},
                    {"index": 1, "relevance_score": 0.1},
                ]
            ),
            None,
        )
    )
    out = reranker.rerank("q", hits, top_n=1)
    assert len(out) == 1
    assert out[0].chunk.text == "a"


def test_ties_preserve_the_response_order_stable_sort():
    hits = [_hit(0, "a"), _hit(1, "b"), _hit(2, "c")]
    reranker, _ = _reranker(
        (
            200,
            _ok(
                [
                    {"index": 1, "relevance_score": 0.5},
                    {"index": 0, "relevance_score": 0.5},
                    {"index": 2, "relevance_score": 0.5},
                ]
            ),
            None,
        )
    )
    out = reranker.rerank("q", hits, top_n=3)
    # Equal scores: Python's stable sort keeps the API's own response order.
    assert [h.chunk.text for h in out] == ["b", "a", "c"]


def test_empty_hits_returns_empty_without_a_request():
    reranker, rec = _reranker((200, _ok([]), None))
    assert reranker.rerank("q", [], top_n=5) == []
    assert rec.requests == []


# --- error mapping (inferred, see module docstring) ----------------------------


def test_401_becomes_rerank_error():
    reranker, _ = _reranker((401, {"detail": "bad key"}, None))
    with pytest.raises(RerankError, match="credentials"):
        reranker.rerank("q", [_hit(0, "a")], top_n=1)


def test_429_is_retried_then_becomes_rerank_error_once_retries_are_spent():
    reranker, rec = _reranker(
        (429, {"detail": "rate limited"}, {"Retry-After": "1"}),
        backoff=Backoff(max_retries=1, base_delay=0.01),
    )
    with pytest.raises(RerankError, match="rate limit"):
        reranker.rerank("q", [_hit(0, "a")], top_n=1)
    assert len(rec.requests) == 2  # one retry


def test_transient_error_is_retried_then_succeeds():
    reranker, rec = _reranker(
        (500, {"detail": "boom"}, None),
        (200, _ok([{"index": 0, "relevance_score": 0.7}]), None),
    )
    out = reranker.rerank("q", [_hit(0, "a")], top_n=1)
    assert [h.score for h in out] == [0.7]
    assert len(rec.requests) == 2


def test_timeout_becomes_rerank_error():
    reranker, _ = _reranker(httpx.TimeoutException("slow"))
    with pytest.raises(RerankError, match="timed out"):
        reranker.rerank("q", [_hit(0, "a")], top_n=1)


def test_connection_failure_becomes_rerank_error():
    reranker, _ = _reranker(httpx.ConnectError("refused"))
    with pytest.raises(RerankError, match="connection failed"):
        reranker.rerank("q", [_hit(0, "a")], top_n=1)


def test_other_4xx_becomes_rerank_error_with_the_body_detail():
    reranker, _ = _reranker((400, {"detail": "bad request shape"}, None))
    with pytest.raises(RerankError, match="bad request shape"):
        reranker.rerank("q", [_hit(0, "a")], top_n=1)


def test_non_json_error_body_falls_back_to_raw_text():
    # A genuinely non-JSON body needs a raw httpx.Response — the Recorder's
    # json= helper always serialises valid JSON.
    def handler(request):
        return httpx.Response(400, text="plain text error")

    reranker = JinaReranker(
        api_key="k", transport=httpx.MockTransport(handler), sleep=lambda s: None
    )
    with pytest.raises(RerankError, match="plain text error"):
        reranker.rerank("q", [_hit(0, "a")], top_n=1)


def test_malformed_response_becomes_rerank_error():
    reranker, _ = _reranker((200, {"no_results_key": True}, None))
    with pytest.raises(RerankError, match="unexpected response shape"):
        reranker.rerank("q", [_hit(0, "a")], top_n=1)


def test_result_index_out_of_range_becomes_rerank_error():
    # A malformed/adversarial response citing an index beyond the request's
    # own document list must not raise an uncaught IndexError.
    reranker, _ = _reranker((200, _ok([{"index": 5, "relevance_score": 0.9}]), None))
    with pytest.raises(RerankError):
        reranker.rerank("q", [_hit(0, "a")], top_n=1)


def test_close_closes_the_http_client():
    reranker, _ = _reranker((200, _ok([]), None))
    reranker.close()
    with pytest.raises(RuntimeError):
        reranker._client.post("/rerank", json={})
