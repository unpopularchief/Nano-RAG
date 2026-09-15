"""``GeminiGenerator`` — the second generation client, for Gemini's own schema.

Gemini does not speak the OpenAI wire format, so it gets its own ~150-line
client (plan.md §4): ``POST /models/{model}:generateContent`` with a
``systemInstruction`` and one user ``content``; the reply is the joined
``candidates[0].content.parts[*].text`` plus ``usageMetadata``.

Error mapping mirrors :mod:`~nanorag.generation.openai_compat`, keyed on
Gemini's ``error.status``: ``UNAUTHENTICATED`` / ``PERMISSION_DENIED`` →
``AuthError``; ``RESOURCE_EXHAUSTED`` (429) → ``QuotaExhausted`` when the
message describes a daily quota, else ``RateLimitError``; ``NOT_FOUND`` →
``ProviderError`` with ``code="model_not_found"``; 5xx / timeouts →
``TransientError``. The key is sent in the ``x-goog-api-key`` header and
never appears in ``repr()`` or an exception.

Google's free tier may train on submitted prompts (plan.md §4,
``docs/providers.md``); the Ollama preset is the private path.
"""

from __future__ import annotations

import os
import random
import time
from collections.abc import Callable
from typing import Any

import httpx

from nanorag.errors import (
    AuthError,
    ConfigError,
    ProviderError,
    QuotaExhausted,
    RateLimitError,
    TransientError,
)
from nanorag.generation.base import (
    Generation,
    looks_like_daily_quota,
    retry_after_seconds,
)
from nanorag.prompting.templates import Prompt
from nanorag.ratelimit import Backoff, RateLimiter, call_with_retry
from nanorag.types import Usage

PROVIDER = "gemini"
API_KEY_ENV = "GEMINI_API_KEY"
BASE_URL = "https://generativelanguage.googleapis.com/v1beta"
#: ``gemini-2.5-flash`` still appears in ``GET /v1beta/models`` but 404s on
#: generation for accounts created after its retirement ("no longer
#: available to new users") — found live at Gate C (2026-09-15), not from
#: the model list, which lags. ``gemini-3.6-flash`` is the model Google's
#: own error message names as the replacement.
DEFAULT_MODEL = "gemini-3.6-flash"
DEFAULT_CONTEXT_WINDOW = 1_048_576
DEFAULT_MAX_OUTPUT_TOKENS = 1024


class GeminiGenerator:
    """Content generation against the Gemini API.

    Parameters
    ----------
    model
        Gemini model name. Defaults to the free-tier Flash model.
    api_key
        Explicit key. Defaults to ``GEMINI_API_KEY``.
    context_window, max_output_tokens
        The model's window and the completion reservation.
    temperature
        Sampling temperature; ``0.0`` for the most reproducible answers.
    timeout, connect_timeout
        Per-request and connection timeouts in seconds.
    backoff, limiter
        Retry schedule and rate limiter (see :mod:`nanorag.ratelimit`).
    sleep, rng, transport
        Injection points for tests.

    Raises
    ------
    ConfigError
        No key was given and ``GEMINI_API_KEY`` is unset.

    """

    provider = PROVIDER

    def __init__(
        self,
        model: str = DEFAULT_MODEL,
        *,
        api_key: str | None = None,
        context_window: int = DEFAULT_CONTEXT_WINDOW,
        max_output_tokens: int = DEFAULT_MAX_OUTPUT_TOKENS,
        temperature: float = 0.0,
        timeout: float = 30.0,
        connect_timeout: float = 5.0,
        backoff: Backoff | None = None,
        limiter: RateLimiter | None = None,
        sleep: Callable[[float], None] = time.sleep,
        rng: random.Random | None = None,
        transport: httpx.BaseTransport | None = None,
        base_url: str = BASE_URL,
    ) -> None:
        """Resolve the key and open an ``httpx.Client`` (see the class docstring)."""
        if api_key is None:
            api_key = os.environ.get(API_KEY_ENV)
            if not api_key:
                raise ConfigError(
                    f"gemini needs an API key: set {API_KEY_ENV}", provider=PROVIDER
                )
        if max_output_tokens < 1:
            raise ConfigError(
                f"max_output_tokens must be >= 1, got {max_output_tokens}"
            )
        self.model = model
        self.context_window = context_window
        self.max_output_tokens = max_output_tokens
        self.temperature = temperature
        self._backoff = backoff if backoff is not None else Backoff()
        self._limiter = limiter
        self._sleep = sleep
        self._rng = rng
        self._client = httpx.Client(
            base_url=base_url,
            headers={"x-goog-api-key": api_key},
            timeout=httpx.Timeout(timeout, connect=connect_timeout),
            transport=transport,
        )

    def __repr__(self) -> str:
        """Return provider and model only — never the key."""
        return f"GeminiGenerator(model={self.model!r})"

    def close(self) -> None:
        """Close the underlying HTTP client."""
        self._client.close()

    def generate(self, prompt: Prompt) -> Generation:
        """Send *prompt* as ``systemInstruction`` + user content; return the text."""
        body = {
            "systemInstruction": {"parts": [{"text": prompt.system}]},
            "contents": [{"role": "user", "parts": [{"text": prompt.user}]}],
            "generationConfig": {
                "temperature": self.temperature,
                "maxOutputTokens": self.max_output_tokens,
            },
        }
        return call_with_retry(
            lambda: self._post(body),
            backoff=self._backoff,
            limiter=self._limiter,
            sleep=self._sleep,
            rng=self._rng,
        )

    def _post(self, body: dict[str, Any]) -> Generation:
        try:
            response = self._client.post(
                f"/models/{self.model}:generateContent", json=body
            )
        except httpx.TimeoutException as exc:
            raise TransientError("request timed out", provider=PROVIDER) from exc
        except httpx.TransportError as exc:
            raise TransientError(
                f"connection failed: {exc}", provider=PROVIDER
            ) from exc
        if response.status_code != 200:
            raise self._error(response)
        try:
            data = response.json()
            parts = data["candidates"][0]["content"]["parts"]
            text = "".join(str(p.get("text", "")) for p in parts)
            usage = data.get("usageMetadata") or {}
        except (ValueError, KeyError, IndexError, TypeError) as exc:
            raise ProviderError("unexpected response shape", provider=PROVIDER) from exc
        return Generation(
            text=text,
            usage=Usage(
                provider=PROVIDER,
                model=str(data.get("modelVersion") or self.model),
                prompt_tokens=int(usage.get("promptTokenCount") or 0),
                completion_tokens=int(usage.get("candidatesTokenCount") or 0),
                total_tokens=int(usage.get("totalTokenCount") or 0),
            ),
        )

    def _error(self, response: httpx.Response) -> ProviderError:
        """Map a non-200 response to the typed error (see the module docstring)."""
        status_code = response.status_code
        message, status = _error_fields(response)
        context: dict[str, object] = {"provider": PROVIDER, "status": status_code}
        if status:
            context["code"] = status
        if status_code in (401, 403) or status in (
            "UNAUTHENTICATED",
            "PERMISSION_DENIED",
        ):
            return AuthError("gemini rejected the credentials", **context)
        if status_code == 429 or status == "RESOURCE_EXHAUSTED":
            if looks_like_daily_quota(message):
                return QuotaExhausted("gemini quota exhausted", **context)
            return RateLimitError(
                "gemini rate limit hit",
                retry_after=retry_after_seconds(response.headers.get("Retry-After")),
                **context,
            )
        if status_code >= 500:
            return TransientError(f"gemini returned {status_code}", **context)
        if status_code == 404 or status == "NOT_FOUND":
            context["code"] = "model_not_found"
            return ProviderError(
                f"gemini does not know model {self.model!r}",
                model=self.model,
                **context,
            )
        return ProviderError(f"gemini returned {status_code}: {message}", **context)


def _error_fields(response: httpx.Response) -> tuple[str, str | None]:
    """Return ``(message, status)`` from a Google-style error body, tolerantly."""
    try:
        error = response.json().get("error")
    except ValueError:
        return response.text[:200], None
    if isinstance(error, dict):
        return str(error.get("message") or ""), error.get("status") or None
    return response.text[:200], None
