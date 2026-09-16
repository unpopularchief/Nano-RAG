import logging

import pytest

from nanorag.errors import (
    AuthError,
    ConfigError,
    EmbeddingError,
    EvaluationError,
    GenerationError,
    IndexModelMismatch,
    LoaderError,
    NanoRagError,
    ProviderError,
    QuotaExhausted,
    RateLimitError,
    RetrievalError,
    StoreError,
    TransientError,
)

ALL_ERRORS = [
    ConfigError,
    LoaderError,
    EmbeddingError,
    StoreError,
    IndexModelMismatch,
    RetrievalError,
    GenerationError,
    EvaluationError,
    ProviderError,
    AuthError,
    RateLimitError,
    TransientError,
    QuotaExhausted,
]


@pytest.mark.parametrize("cls", ALL_ERRORS)
def test_every_error_is_a_nanorag_error(cls):
    assert issubclass(cls, NanoRagError)
    with pytest.raises(NanoRagError):
        raise cls("boom")


@pytest.mark.parametrize(
    "cls", [AuthError, RateLimitError, TransientError, QuotaExhausted]
)
def test_provider_error_subclasses(cls):
    assert issubclass(cls, ProviderError)


def test_index_model_mismatch_is_a_store_error():
    assert issubclass(IndexModelMismatch, StoreError)


def test_provider_error_is_not_a_generation_error():
    # Hosted embedding backends raise ProviderError too, so it sits under
    # NanoRagError directly, not under GenerationError.
    assert not issubclass(ProviderError, GenerationError)


def test_message_and_context_are_accessible():
    err = LoaderError("could not read", path="/data/x.md", encoding="latin-1")
    assert err.message == "could not read"
    assert err.context == {"path": "/data/x.md", "encoding": "latin-1"}


def test_non_secret_context_is_shown_in_str():
    err = LoaderError("nope", path="/data/x.md")
    assert "/data/x.md" in str(err)
    assert str(err).startswith("nope")


@pytest.mark.parametrize(
    "key",
    [
        "api_key",
        "GROQ_API_KEY",
        "authorization",
        "auth_token",
        "session_cookie",
        "password",
    ],
)
def test_secret_looking_context_keys_are_redacted(key):
    secret = "sk-do-not-leak-1234567890"
    err = AuthError("rejected", **{key: secret})
    assert secret not in str(err)
    assert secret not in repr(err)
    assert "***redacted***" in str(err)


def test_secret_is_absent_from_logged_record(caplog):
    secret = "sk-super-secret-value"
    with caplog.at_level(logging.ERROR):
        try:
            raise ProviderError("call failed", GROQ_API_KEY=secret)
        except ProviderError as exc:
            logging.getLogger("test").error("generation failed: %s", exc)
    assert secret not in caplog.text
    assert "***redacted***" in caplog.text


def test_rate_limit_error_carries_retry_after():
    err = RateLimitError("slow down", retry_after=2.5)
    assert err.retry_after == 2.5
    assert "retry_after" in str(err)
    assert isinstance(err, ProviderError)


def test_rate_limit_error_retry_after_defaults_to_none():
    err = RateLimitError("slow down")
    assert err.retry_after is None
    assert "retry_after" not in str(err)


def test_str_without_context_is_just_the_message():
    assert str(ConfigError("missing extra: local")) == "missing extra: local"


def test_catching_base_catches_provider_errors():
    for cls in (AuthError, RateLimitError, TransientError, QuotaExhausted):
        try:
            raise cls("x")
        except NanoRagError:
            pass
        else:  # pragma: no cover
            pytest.fail(f"{cls.__name__} not caught by NanoRagError")
