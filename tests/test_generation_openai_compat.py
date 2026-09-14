"""``OpenAICompatGenerator`` against recorded responses via ``MockTransport``:
request shape, usage parsing, the full error mapping (a real Groq 429 body
with ``Retry-After``, a daily-cap 429, ``model_not_found``, 401, 5xx,
timeouts), retry behaviour, and that the key never leaks."""

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
from nanorag.generation import GROQ, OLLAMA, OpenAICompatGenerator, Preset
from nanorag.generation.presets import PRESETS
from nanorag.prompting.templates import Prompt
from nanorag.ratelimit import Backoff, RateLimiter
from tests.fakes import FakeClock

PROMPT = Prompt(system="be brief {n}", user="ctx\n\nQuestion: why?", nonce="n")

# A real Groq per-minute 429 body (organisation id elided).
GROQ_429_TPM = {
    "error": {
        "message": (
            "Rate limit reached for model `llama-3.3-70b-versatile` in organization "
            "`org_x` service tier `on_demand` on tokens per minute (TPM): Limit "
            "6000, Used 0, Requested 8000. Please try again in 20s. Need more "
            "tokens? Upgrade to Dev Tier today at https://console.groq.com/settings/billing"
        ),
        "type": "tokens",
        "code": "rate_limit_exceeded",
    }
}
# The same, for the daily request cap.
GROQ_429_RPD = {
    "error": {
        "message": (
            "Rate limit reached for model `llama-3.3-70b-versatile` in organization "
            "`org_x` service tier `on_demand` on requests per day (RPD): Limit 1000, "
            "Used 1000, Requested 1. Please try again in 2h3m."
        ),
        "type": "requests",
        "code": "rate_limit_exceeded",
    }
}
GROQ_404_MODEL = {
    "error": {
        "message": (
            "The model `llama3-70b-8192` does not exist or you do not have "
            "access to it."
        ),
        "type": "invalid_request_error",
        "code": "model_not_found",
    }
}


def _ok(text="Answer [1]", model="llama-3.3-70b-versatile"):
    return {
        "id": "chatcmpl-x",
        "model": model,
        "choices": [{"index": 0, "message": {"role": "assistant", "content": text}}],
        "usage": {"prompt_tokens": 40, "completion_tokens": 5, "total_tokens": 45},
    }


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


def _gen(*responses, preset=GROQ, **kwargs):
    recorder = Recorder(*responses)
    kwargs.setdefault("api_key", "sk-test-secret")
    kwargs.setdefault("backoff", Backoff(max_retries=2, base_delay=0.01))
    kwargs.setdefault("sleep", lambda s: None)
    gen = OpenAICompatGenerator(
        preset, transport=httpx.MockTransport(recorder), **kwargs
    )
    return gen, recorder


# --- construction -------------------------------------------------------------


def test_key_comes_from_the_preset_env_var(monkeypatch):
    monkeypatch.delenv("GROQ_API_KEY", raising=False)
    with pytest.raises(ConfigError, match="GROQ_API_KEY"):
        OpenAICompatGenerator(GROQ)
    monkeypatch.setenv("GROQ_API_KEY", "from-env")
    gen = OpenAICompatGenerator(GROQ, transport=httpx.MockTransport(Recorder()))
    assert gen._client.headers["Authorization"] == "Bearer from-env"


def test_keyless_preset_needs_no_key(monkeypatch):
    monkeypatch.delenv("GROQ_API_KEY", raising=False)
    gen = OpenAICompatGenerator(OLLAMA, transport=httpx.MockTransport(Recorder()))
    assert "Authorization" not in gen._client.headers
    assert gen.provider == "ollama"
    assert gen.model == OLLAMA.default_model
    assert gen.context_window == 4096


def test_defaults_come_from_the_preset_and_are_overridable():
    gen, _ = _gen((200, _ok(), None))
    assert (gen.provider, gen.model, gen.context_window) == (
        "groq",
        "llama-3.3-70b-versatile",
        131_072,
    )
    gen2, _ = _gen((200, _ok(), None), model="other", context_window=8192)
    assert (gen2.model, gen2.context_window) == ("other", 8192)


def test_max_output_tokens_must_be_positive():
    with pytest.raises(ConfigError):
        _gen((200, _ok(), None), max_output_tokens=0)


def test_repr_and_errors_never_contain_the_key():
    gen, _ = _gen((401, {"error": {"message": "bad key"}}, None))
    assert "sk-test-secret" not in repr(gen)
    with pytest.raises(AuthError) as info:
        gen.generate(PROMPT)
    assert "sk-test-secret" not in str(info.value)
    assert "sk-test-secret" not in repr(info.value)


def test_presets_table_and_validation():
    assert set(PRESETS) == {"groq", "openrouter", "ollama"}
    with pytest.raises(ConfigError):
        Preset("x", "http://x", None, "m", 0)
    with pytest.raises(ConfigError):
        Preset("", "http://x", None, "m", 1)


# --- the happy path -----------------------------------------------------------


def test_sends_system_and_user_messages_and_parses_usage():
    gen, rec = _gen((200, _ok("The cat sat. [1]"), None), max_output_tokens=77)
    out = gen.generate(PROMPT)

    assert out.text == "The cat sat. [1]"
    assert out.usage.provider == "groq"
    assert out.usage.model == "llama-3.3-70b-versatile"
    assert (out.usage.prompt_tokens, out.usage.completion_tokens) == (40, 5)
    assert out.usage.total_tokens == 45

    request = rec.requests[0]
    assert request.method == "POST"
    assert str(request.url) == "https://api.groq.com/openai/v1/chat/completions"
    assert request.headers["Authorization"] == "Bearer sk-test-secret"
    body = json.loads(request.content)
    assert body["messages"] == [
        {"role": "system", "content": PROMPT.system},
        {"role": "user", "content": PROMPT.user},
    ]
    assert body["model"] == "llama-3.3-70b-versatile"
    assert body["max_tokens"] == 77
    assert body["temperature"] == 0.0


def test_missing_usage_and_null_content_are_tolerated():
    body = _ok()
    body["choices"][0]["message"]["content"] = None
    del body["usage"]
    gen, _ = _gen((200, body, None))
    out = gen.generate(PROMPT)
    assert out.text == ""
    assert out.usage.total_tokens == 0


@pytest.mark.parametrize("body", [{"choices": []}, {"nope": 1}, "not json"])
def test_unexpected_response_shape_is_a_provider_error(body):
    def handler(request):
        if isinstance(body, str):
            return httpx.Response(200, text=body)
        return httpx.Response(200, json=body)

    gen = OpenAICompatGenerator(
        GROQ, api_key="k", transport=httpx.MockTransport(handler)
    )
    with pytest.raises(ProviderError):
        gen.generate(PROMPT)


# --- error mapping ------------------------------------------------------------


def test_real_groq_429_with_retry_after_is_retried_then_raised():
    clock = FakeClock()
    gen, rec = _gen(
        (429, GROQ_429_TPM, {"Retry-After": "20"}),
        backoff=Backoff(max_retries=2, base_delay=0.01),
        sleep=clock.sleep,
    )
    with pytest.raises(RateLimitError) as info:
        gen.generate(PROMPT)
    assert info.value.retry_after == 20.0
    assert info.value.context["status"] == 429
    assert info.value.context["code"] == "rate_limit_exceeded"
    assert len(rec.requests) == 3  # first attempt + 2 retries
    assert clock.sleeps == [20.0, 20.0]  # Retry-After honoured verbatim


def test_429_recovers_when_a_retry_succeeds():
    gen, rec = _gen(
        (429, GROQ_429_TPM, {"Retry-After": "1"}),
        (200, _ok("recovered"), None),
    )
    assert gen.generate(PROMPT).text == "recovered"
    assert len(rec.requests) == 2


def test_daily_cap_429_is_quota_exhausted_and_not_retried():
    gen, rec = _gen((429, GROQ_429_RPD, {"Retry-After": "7380"}))
    with pytest.raises(QuotaExhausted):
        gen.generate(PROMPT)
    assert len(rec.requests) == 1


def test_retired_model_name_is_a_provider_error_tagged_model_not_found():
    gen, rec = _gen((404, GROQ_404_MODEL, None), model="llama3-70b-8192")
    with pytest.raises(ProviderError) as info:
        gen.generate(PROMPT)
    assert not isinstance(info.value, RateLimitError | TransientError | AuthError)
    assert info.value.context["code"] == "model_not_found"
    assert info.value.context["model"] == "llama3-70b-8192"
    assert len(rec.requests) == 1


@pytest.mark.parametrize("status", [401, 403])
def test_auth_failures_are_auth_errors_never_retried(status):
    gen, rec = _gen((status, {"error": {"message": "Invalid API Key"}}, None))
    with pytest.raises(AuthError):
        gen.generate(PROMPT)
    assert len(rec.requests) == 1


@pytest.mark.parametrize("status", [500, 502, 503])
def test_5xx_is_transient_and_retried(status):
    gen, rec = _gen((status, {"error": "upstream"}, None))
    with pytest.raises(TransientError):
        gen.generate(PROMPT)
    assert len(rec.requests) == 3


def test_5xx_with_a_non_json_body_is_still_transient():
    gen = OpenAICompatGenerator(
        GROQ,
        api_key="k",
        backoff=Backoff(max_retries=0),
        transport=httpx.MockTransport(lambda r: httpx.Response(503, text="<html>")),
    )
    with pytest.raises(TransientError):
        gen.generate(PROMPT)


def test_timeout_and_connection_failure_are_transient():
    for exc in (httpx.ReadTimeout("slow"), httpx.ConnectError("refused")):
        gen, rec = _gen(exc, backoff=Backoff(max_retries=1, base_delay=0.01))
        with pytest.raises(TransientError):
            gen.generate(PROMPT)
        assert len(rec.requests) == 2


def test_other_4xx_is_a_provider_error_not_retried():
    gen, rec = _gen((400, {"error": {"message": "bad request"}}, None))
    with pytest.raises(ProviderError, match="bad request") as info:
        gen.generate(PROMPT)
    assert type(info.value) is ProviderError
    assert len(rec.requests) == 1


def test_string_error_body_is_used_as_the_message():
    gen, _ = _gen((400, {"error": "plain string"}, None))
    with pytest.raises(ProviderError, match="plain string"):
        gen.generate(PROMPT)


# --- rate limiter integration -------------------------------------------------


def test_limiter_gates_requests_and_absorbs_retry_after():
    clock = FakeClock()
    limiter = RateLimiter(rpm=30, clock=clock.monotonic, sleep=clock.sleep)
    gen, rec = _gen(
        (429, GROQ_429_TPM, {"Retry-After": "5"}),
        (200, _ok(), None),
        limiter=limiter,
        sleep=clock.sleep,
    )
    gen.generate(PROMPT)
    assert clock.sleeps[0] == 5.0  # the retry slept Retry-After
    assert len(rec.requests) == 2


def test_close_closes_the_client():
    gen, _ = _gen((200, _ok(), None))
    gen.close()
    assert gen._client.is_closed


def test_json_error_body_that_is_neither_dict_nor_string_falls_back_to_text():
    gen, _ = _gen((400, {"error": ["weird", "shape"]}, None))
    with pytest.raises(ProviderError, match="weird"):
        gen.generate(PROMPT)
