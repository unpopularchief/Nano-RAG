"""``GeminiGenerator`` against recorded responses via ``MockTransport``:
Gemini's own request/response schema, ``RESOURCE_EXHAUSTED`` split into
rate-limit vs quota, ``NOT_FOUND`` as model-not-found, auth, 5xx, timeouts,
and the shared 429 / Retry-After helpers."""

import json

import httpx
import pytest

from nanorag.errors import (
    AuthError,
    ConfigError,
    ProviderError,
    QuotaExhausted,
    RateLimitError,
    TransientError,
)
from nanorag.generation import (
    GeminiGenerator,
    looks_like_daily_quota,
    retry_after_seconds,
)
from nanorag.prompting.templates import Prompt
from nanorag.ratelimit import Backoff

PROMPT = Prompt(system="rules {n}", user="ctx\n\nQuestion: why?", nonce="n")

GEMINI_OK = {
    "candidates": [
        {
            "content": {
                "parts": [{"text": "The cat "}, {"text": "sat. [1]"}],
                "role": "model",
            },
            "finishReason": "STOP",
        }
    ],
    "usageMetadata": {
        "promptTokenCount": 33,
        "candidatesTokenCount": 6,
        "totalTokenCount": 39,
    },
    "modelVersion": "gemini-2.5-flash",
}
GEMINI_429_RPM = {
    "error": {
        "code": 429,
        "message": "Resource has been exhausted (e.g. check quota).",
        "status": "RESOURCE_EXHAUSTED",
    }
}
GEMINI_429_DAILY = {
    "error": {
        "code": 429,
        "message": (
            "You exceeded your current quota, please check your plan and billing "
            "details. Quota exceeded for quota metric 'Generate Content API "
            "requests per day' and limit 'GenerateRequestsPerDayPerProjectPerModel"
            "-FreeTier' of service 'generativelanguage.googleapis.com'."
        ),
        "status": "RESOURCE_EXHAUSTED",
    }
}
GEMINI_404 = {
    "error": {
        "code": 404,
        "message": (
            "models/gemini-1.0-pro is not found for API version v1beta, or is "
            "not supported for generateContent."
        ),
        "status": "NOT_FOUND",
    }
}


class Recorder:
    def __init__(self, *responses):
        self.responses = list(responses)
        self.requests: list[httpx.Request] = []

    def __call__(self, request):
        self.requests.append(request)
        item = self.responses.pop(0) if len(self.responses) > 1 else self.responses[0]
        if isinstance(item, Exception):
            raise item
        status, body, headers = item
        return httpx.Response(status, json=body, headers=headers or {})


def _gen(*responses, **kwargs):
    recorder = Recorder(*responses)
    kwargs.setdefault("api_key", "AIza-secret")
    kwargs.setdefault("backoff", Backoff(max_retries=2, base_delay=0.01))
    kwargs.setdefault("sleep", lambda s: None)
    return GeminiGenerator(transport=httpx.MockTransport(recorder), **kwargs), recorder


# --- construction -------------------------------------------------------------


def test_key_comes_from_gemini_api_key(monkeypatch):
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    with pytest.raises(ConfigError, match="GEMINI_API_KEY"):
        GeminiGenerator()
    monkeypatch.setenv("GEMINI_API_KEY", "from-env")
    gen = GeminiGenerator(transport=httpx.MockTransport(Recorder()))
    assert gen._client.headers["x-goog-api-key"] == "from-env"


def test_defaults_and_repr_without_key():
    gen, _ = _gen((200, GEMINI_OK, None))
    assert gen.provider == "gemini"
    assert gen.model == "gemini-2.5-flash"
    assert gen.context_window == 1_048_576
    assert gen.max_output_tokens == 1024
    assert "AIza-secret" not in repr(gen)
    with pytest.raises(ConfigError):
        _gen((200, GEMINI_OK, None), max_output_tokens=0)


# --- the happy path -----------------------------------------------------------


def test_sends_system_instruction_and_user_content_and_joins_parts():
    gen, rec = _gen((200, GEMINI_OK, None), max_output_tokens=50, temperature=0.2)
    out = gen.generate(PROMPT)

    assert out.text == "The cat sat. [1]"
    assert out.usage.provider == "gemini"
    assert out.usage.model == "gemini-2.5-flash"
    assert (out.usage.prompt_tokens, out.usage.completion_tokens) == (33, 6)
    assert out.usage.total_tokens == 39

    request = rec.requests[0]
    assert request.method == "POST"
    assert str(request.url) == (
        "https://generativelanguage.googleapis.com/v1beta/models/"
        "gemini-2.5-flash:generateContent"
    )
    assert request.headers["x-goog-api-key"] == "AIza-secret"
    body = json.loads(request.content)
    assert body["systemInstruction"] == {"parts": [{"text": PROMPT.system}]}
    assert body["contents"] == [{"role": "user", "parts": [{"text": PROMPT.user}]}]
    assert body["generationConfig"] == {"temperature": 0.2, "maxOutputTokens": 50}


def test_missing_usage_metadata_is_tolerated():
    body = {"candidates": [{"content": {"parts": [{"text": "x"}]}}]}
    gen, _ = _gen((200, body, None))
    out = gen.generate(PROMPT)
    assert out.text == "x"
    assert out.usage.total_tokens == 0
    assert out.usage.model == "gemini-2.5-flash"


@pytest.mark.parametrize("body", [{"candidates": []}, {"promptFeedback": {}}])
def test_unexpected_shape_is_a_provider_error(body):
    gen, _ = _gen((200, body, None))
    with pytest.raises(ProviderError):
        gen.generate(PROMPT)


# --- error mapping ------------------------------------------------------------


def test_resource_exhausted_without_quota_wording_is_a_rate_limit_retried():
    gen, rec = _gen((429, GEMINI_429_RPM, {"Retry-After": "3"}))
    with pytest.raises(RateLimitError) as info:
        gen.generate(PROMPT)
    assert info.value.retry_after == 3.0
    assert info.value.context["code"] == "RESOURCE_EXHAUSTED"
    assert len(rec.requests) == 3


def test_daily_quota_wording_is_quota_exhausted_not_retried():
    gen, rec = _gen((429, GEMINI_429_DAILY, None))
    with pytest.raises(QuotaExhausted):
        gen.generate(PROMPT)
    assert len(rec.requests) == 1


def test_not_found_is_model_not_found():
    gen, rec = _gen((404, GEMINI_404, None), model="gemini-1.0-pro")
    with pytest.raises(ProviderError) as info:
        gen.generate(PROMPT)
    assert type(info.value) is ProviderError
    assert info.value.context["code"] == "model_not_found"
    assert info.value.context["model"] == "gemini-1.0-pro"
    assert len(rec.requests) == 1


@pytest.mark.parametrize(
    "status,body",
    [
        (401, {"error": {"code": 401, "status": "UNAUTHENTICATED", "message": "x"}}),
        (403, {"error": {"code": 403, "status": "PERMISSION_DENIED", "message": "x"}}),
        (400, {"error": {"code": 400, "status": "UNAUTHENTICATED", "message": "x"}}),
    ],
)
def test_auth_failures_by_status_code_or_status_string(status, body):
    gen, rec = _gen((status, body, None))
    with pytest.raises(AuthError) as info:
        gen.generate(PROMPT)
    assert "AIza-secret" not in str(info.value)
    assert len(rec.requests) == 1


def test_5xx_timeouts_and_connection_failures_are_transient():
    for item in (
        (503, {"error": "x"}, None),
        httpx.ReadTimeout("t"),
        httpx.ConnectError("c"),
    ):
        gen, rec = _gen(item, backoff=Backoff(max_retries=1, base_delay=0.01))
        with pytest.raises(TransientError):
            gen.generate(PROMPT)
        assert len(rec.requests) == 2


def test_other_4xx_is_a_provider_error_with_the_message():
    gen, _ = _gen(
        (400, {"error": {"code": 400, "message": "INVALID_ARGUMENT x"}}, None)
    )
    with pytest.raises(ProviderError, match="INVALID_ARGUMENT x"):
        gen.generate(PROMPT)


def test_non_json_error_body_is_tolerated():
    gen = GeminiGenerator(
        api_key="k",
        backoff=Backoff(max_retries=0),
        transport=httpx.MockTransport(lambda r: httpx.Response(400, text="<html>")),
    )
    with pytest.raises(ProviderError, match="<html>"):
        gen.generate(PROMPT)


# --- shared helpers -----------------------------------------------------------


@pytest.mark.parametrize(
    "message,expected",
    [
        ("on requests per day (RPD): Limit 1000", True),
        ("on tokens per minute (TPM): Limit 6000", False),
        ("You exceeded your current quota", False),
        ("Resource has been exhausted (e.g. check quota).", False),
        ("limit 'GenerateRequestsPerDayPerProjectPerModel-FreeTier'", True),
        ("limit 'GenerateRequestsPerMinutePerProjectPerModel-FreeTier'", False),
        ("Rate limit reached", False),
        ("", False),
    ],
)
def test_looks_like_daily_quota(message, expected):
    assert looks_like_daily_quota(message) is expected


@pytest.mark.parametrize(
    "value,expected",
    [
        (None, None),
        ("20", 20.0),
        (" 1.5 ", 1.5),
        ("0", 0.0),
        ("-3", None),
        ("Wed, 21 Oct 2015 07:28:00 GMT", None),
    ],
)
def test_retry_after_seconds(value, expected):
    assert retry_after_seconds(value) == expected


def test_close_closes_the_client():
    gen, _ = _gen((200, GEMINI_OK, None))
    gen.close()
    assert gen._client.is_closed
