"""``OpenAICompatGenerator`` — one client for every OpenAI-wire-format service.

Groq, OpenRouter and Ollama (plus Together, LM Studio, vLLM, …) differ only
in ``base_url``, key and model, all carried by a
:class:`~nanorag.generation.presets.Preset`.
The request is a single ``POST /chat/completions``; the response is
``choices[0].message.content`` plus ``usage``.

Error mapping (plan.md §11 "Provider failures"), by HTTP status:

- 401 / 403 → ``AuthError`` — never retried.
- 404 with an error ``code`` of ``model_not_found`` → ``ProviderError`` with
  ``code="model_not_found"`` in its context. Groq retires model names
  often; the fallback chain treats this exactly like any other provider
  failure (plan.md §18 F6).
- 429 → ``QuotaExhausted`` when the body describes a daily cap (Groq
  reports RPD/TPD exhaustion with the same status as a per-minute limit),
  else ``RateLimitError`` carrying ``Retry-After``. Retried with backoff.
- 5xx, timeouts, connection failures → ``TransientError``. Retried.
- Any other 4xx, or a body that does not parse → ``ProviderError``. Not
  retried.

The key is read from the preset's environment variable, held privately and
never appears in ``repr()``, log output or an exception (plan.md §13).
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
from nanorag.generation.presets import GROQ, Preset
from nanorag.prompting.templates import Prompt
from nanorag.ratelimit import Backoff, RateLimiter, call_with_retry
from nanorag.types import Usage

#: Default completion reservation, in tokens.
DEFAULT_MAX_OUTPUT_TOKENS = 1024


class OpenAICompatGenerator:
    """Chat completions against any OpenAI-compatible endpoint.

    Parameters
    ----------
    preset
        Which service. Defaults to Groq (plan.md §4 primary).
    model
        Model name; defaults to the preset's.
    api_key
        Explicit key. Defaults to the preset's environment variable; a
        preset with ``api_key_env=None`` (Ollama) needs none.
    context_window, max_output_tokens
        The model's window and the completion reservation — what the
        pipeline budgets against. ``context_window`` defaults to the
        preset's.
    temperature
        Sampling temperature; ``0.0`` for the most reproducible answers.
    timeout, connect_timeout
        Per-request and connection timeouts in seconds.
    backoff, limiter
        Retry schedule and rate limiter (see :mod:`nanorag.ratelimit`).
    sleep, rng, transport
        Injection points for tests: the sleep used between retries, the
        jitter source, and an ``httpx`` transport (``MockTransport``).

    Raises
    ------
    ConfigError
        The preset needs a key and none was given or found in the
        environment.

    """

    def __init__(
        self,
        preset: Preset = GROQ,
        model: str | None = None,
        *,
        api_key: str | None = None,
        context_window: int | None = None,
        max_output_tokens: int = DEFAULT_MAX_OUTPUT_TOKENS,
        temperature: float = 0.0,
        timeout: float = 30.0,
        connect_timeout: float = 5.0,
        backoff: Backoff | None = None,
        limiter: RateLimiter | None = None,
        sleep: Callable[[float], None] = time.sleep,
        rng: random.Random | None = None,
        transport: httpx.BaseTransport | None = None,
    ) -> None:
        """Resolve the key and open an ``httpx.Client`` (see the class docstring)."""
        if preset.api_key_env is not None and api_key is None:
            api_key = os.environ.get(preset.api_key_env)
            if not api_key:
                raise ConfigError(
                    f"{preset.name} needs an API key: set {preset.api_key_env}",
                    provider=preset.name,
                )
        if max_output_tokens < 1:
            raise ConfigError(
                f"max_output_tokens must be >= 1, got {max_output_tokens}"
            )
        self.preset = preset
        self.provider = preset.name
        self.model = model or preset.default_model
        self.context_window = context_window or preset.context_window
        self.max_output_tokens = max_output_tokens
        self.temperature = temperature
        self._backoff = backoff if backoff is not None else Backoff()
        self._limiter = limiter
        self._sleep = sleep
        self._rng = rng
        headers = {"Authorization": f"Bearer {api_key}"} if api_key else {}
        self._client = httpx.Client(
            base_url=preset.base_url,
            headers=headers,
            timeout=httpx.Timeout(timeout, connect=connect_timeout),
            transport=transport,
        )

    def __repr__(self) -> str:
        """Return provider and model only — never the key."""
        return (
            f"OpenAICompatGenerator(provider={self.provider!r}, model={self.model!r})"
        )

    def close(self) -> None:
        """Close the underlying HTTP client."""
        self._client.close()

    def generate(self, prompt: Prompt) -> Generation:
        """Send *prompt* as a system + user chat and return the completion.

        Retries on ``RateLimitError`` / ``TransientError`` per the backoff;
        every other error is raised on the first occurrence.
        """
        body = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": prompt.system},
                {"role": "user", "content": prompt.user},
            ],
            "temperature": self.temperature,
            "max_tokens": self.max_output_tokens,
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
            response = self._client.post("/chat/completions", json=body)
        except httpx.TimeoutException as exc:
            raise TransientError("request timed out", provider=self.provider) from exc
        except httpx.TransportError as exc:
            raise TransientError(
                f"connection failed: {exc}", provider=self.provider
            ) from exc
        if response.status_code != 200:
            raise self._error(response)
        try:
            data = response.json()
            text = data["choices"][0]["message"]["content"]
            usage = data.get("usage") or {}
        except (ValueError, KeyError, IndexError, TypeError) as exc:
            raise ProviderError(
                "unexpected response shape", provider=self.provider
            ) from exc
        return Generation(
            text=text or "",
            usage=Usage(
                provider=self.provider,
                model=str(data.get("model") or self.model),
                prompt_tokens=int(usage.get("prompt_tokens") or 0),
                completion_tokens=int(usage.get("completion_tokens") or 0),
                total_tokens=int(usage.get("total_tokens") or 0),
            ),
        )

    def _error(self, response: httpx.Response) -> ProviderError:
        """Map a non-200 response to the typed error (see the module docstring)."""
        status = response.status_code
        message, code = _error_fields(response)
        context: dict[str, object] = {"provider": self.provider, "status": status}
        if code:
            context["code"] = code
        if status in (401, 403):
            return AuthError(f"{self.provider} rejected the credentials", **context)
        if status == 429:
            if looks_like_daily_quota(message):
                return QuotaExhausted(f"{self.provider} quota exhausted", **context)
            return RateLimitError(
                f"{self.provider} rate limit hit",
                retry_after=retry_after_seconds(response.headers.get("Retry-After")),
                **context,
            )
        if status >= 500:
            return TransientError(f"{self.provider} returned {status}", **context)
        if status == 404 or code == "model_not_found":
            context["code"] = "model_not_found"
            return ProviderError(
                f"{self.provider} does not know model {self.model!r}",
                model=self.model,
                **context,
            )
        return ProviderError(f"{self.provider} returned {status}: {message}", **context)


def _error_fields(response: httpx.Response) -> tuple[str, str | None]:
    """Return ``(message, code)`` from an OpenAI-style error body, tolerantly."""
    try:
        error = response.json().get("error")
    except ValueError:
        return response.text[:200], None
    if isinstance(error, dict):
        return str(error.get("message") or ""), error.get("code") or None
    if isinstance(error, str):
        return error, None
    return response.text[:200], None
