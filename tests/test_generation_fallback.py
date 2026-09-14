"""``FallbackGenerator`` (plan.md §18 F6): quota exhaustion, a provider error
and a retired model name each fall back from the primary to the next
provider and record which one served; a rejected key does not; the tightest
window wins; every hand-off is logged. Also the ``Generation`` value type."""

import json
import logging

import httpx
import pytest

from nanorag.errors import (
    AuthError,
    ProviderError,
    QuotaExhausted,
    RateLimitError,
    TransientError,
)
from nanorag.generation import (
    GROQ,
    FallbackGenerator,
    GeminiGenerator,
    Generation,
    Generator,
    OpenAICompatGenerator,
)
from nanorag.prompting.templates import Prompt
from nanorag.ratelimit import Backoff
from nanorag.types import Usage
from tests.fakes import FakeGenerator
from tests.test_generation_gemini import GEMINI_OK
from tests.test_generation_openai_compat import (
    GROQ_404_MODEL,
    GROQ_429_RPD,
    Recorder,
)

PROMPT = Prompt(system="s {n}", user="u", nonce="n")


def _fake(provider, *responses, **kwargs):
    return FakeGenerator(
        list(responses), provider=provider, model=f"{provider}-m", **kwargs
    )


# --- chain semantics ----------------------------------------------------------


def test_primary_answer_is_returned_untouched():
    primary, backup = _fake("groq", "from groq"), _fake("gemini", "from gemini")
    chain = FallbackGenerator(primary, backup)
    out = chain.generate(PROMPT)
    assert out.text == "from groq"
    assert out.usage.provider == "groq"
    assert backup.calls == []
    assert isinstance(chain, Generator)
    assert (chain.provider, chain.model) == ("groq", "groq-m")


@pytest.mark.parametrize(
    "failure",
    [
        QuotaExhausted("daily cap spent"),
        ProviderError("does not know model", code="model_not_found"),
        ProviderError("400 something else"),
        RateLimitError("still 429 after retries", retry_after=5.0),
        TransientError("still 503 after retries"),
    ],
)
def test_falls_back_and_records_the_serving_provider(failure, caplog):
    primary, backup = _fake("groq", failure), _fake("gemini", "from gemini")
    chain = FallbackGenerator(primary, backup)
    with caplog.at_level(logging.WARNING, logger="nanorag.generation"):
        out = chain.generate(PROMPT)
    assert out.text == "from gemini"
    assert out.usage.provider == "gemini"
    assert len(primary.calls) == 1 and len(backup.calls) == 1
    assert "falling back to gemini" in caplog.text
    assert type(failure).__name__ in caplog.text


def test_auth_error_is_raised_immediately_without_fallback():
    primary, backup = _fake("groq", AuthError("bad key")), _fake("gemini", "x")
    with pytest.raises(AuthError):
        FallbackGenerator(primary, backup).generate(PROMPT)
    assert backup.calls == []


def test_all_members_failing_raises_the_last_error():
    chain = FallbackGenerator(
        _fake("groq", QuotaExhausted("groq spent")),
        _fake("gemini", QuotaExhausted("gemini spent")),
        _fake("ollama", TransientError("ollama down")),
    )
    with pytest.raises(TransientError, match="ollama down"):
        chain.generate(PROMPT)


def test_non_provider_errors_propagate_untouched():
    chain = FallbackGenerator(_fake("groq", ValueError("bug")), _fake("gemini", "x"))
    with pytest.raises(ValueError):
        chain.generate(PROMPT)


def test_third_member_serves_when_first_two_fail():
    chain = FallbackGenerator(
        _fake("groq", QuotaExhausted("a")),
        _fake("gemini", ProviderError("b")),
        _fake("ollama", "from ollama"),
    )
    assert chain.generate(PROMPT).usage.provider == "ollama"


def test_window_and_reservation_are_the_minimum_across_the_chain():
    chain = FallbackGenerator(
        _fake("groq", context_window=131_072, max_output_tokens=1024),
        _fake("ollama", context_window=4096, max_output_tokens=512),
        _fake("gemini", context_window=1_000_000, max_output_tokens=2048),
    )
    assert chain.context_window == 4096
    assert chain.max_output_tokens == 512
    assert repr(chain) == "FallbackGenerator(groq -> ollama -> gemini)"


def test_single_member_chain_is_just_that_member():
    only = _fake("groq", "solo")
    assert FallbackGenerator(only).generate(PROMPT).text == "solo"


# --- real clients in the chain, over MockTransport ----------------------------


@pytest.mark.parametrize(
    "groq_response",
    [
        (429, GROQ_429_RPD, {"Retry-After": "7380"}),  # daily quota
        (404, GROQ_404_MODEL, None),  # retired model name
        (400, {"error": {"message": "invalid request"}}, None),  # provider error
    ],
)
def test_groq_to_gemini_handoff_over_http(groq_response):
    groq_rec = Recorder(groq_response)
    gemini_rec = Recorder((200, GEMINI_OK, None))
    groq = OpenAICompatGenerator(
        GROQ,
        api_key="g",
        backoff=Backoff(max_retries=0),
        transport=httpx.MockTransport(groq_rec),
    )
    gemini = GeminiGenerator(
        api_key="k",
        backoff=Backoff(max_retries=0),
        transport=httpx.MockTransport(gemini_rec),
    )
    out = FallbackGenerator(groq, gemini).generate(PROMPT)
    assert out.text == "The cat sat. [1]"
    assert out.usage.provider == "gemini"
    assert len(groq_rec.requests) == 1 and len(gemini_rec.requests) == 1
    # The same prompt reached both providers.
    assert json.loads(gemini_rec.requests[0].content)["contents"][0]["parts"] == [
        {"text": PROMPT.user}
    ]


# --- Generation ---------------------------------------------------------------


def test_generation_is_frozen_and_serialisable():
    gen = Generation(text="t", usage=Usage("p", "m", 1, 2, 3))
    with pytest.raises(AttributeError):
        gen.text = "x"  # type: ignore[misc]
    assert json.loads(json.dumps(gen.to_dict())) == {
        "text": "t",
        "usage": {
            "provider": "p",
            "model": "m",
            "prompt_tokens": 1,
            "completion_tokens": 2,
            "total_tokens": 3,
            "cost_usd": 0.0,
        },
    }
