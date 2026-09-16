"""``JinaReranker`` — cross-encoder reranking via the Jina Reranker API.

plan.md §4: 1M tokens free, 100 RPM, a non-commercial key
(``docs/providers.md`` states the terms). One HTTP call per query:
``POST /v1/rerank`` with the query and every candidate's chunk text; the
response is a list of ``{index, relevance_score}`` pairs, one per
candidate, already sorted best-first.

Error mapping mirrors :mod:`~nanorag.generation.openai_compat` (401/403 →
``AuthError``, 429 → ``RateLimitError``, 5xx/timeout → ``TransientError``,
other 4xx → ``ProviderError``) but nothing here has been observed against a
live key — no ``JINA_API_KEY`` was available this session (docs/evaluation.md
notes it as a named follow-up), so unlike Groq/Gemini's mapping this one is
inferred from the general Jina API shape, not confirmed live traffic. Any
``ProviderError`` this raises internally is translated to ``RerankError``
at the ``rerank()`` boundary — the one error type the ``Reranker`` protocol
promises its caller.
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
    RateLimitError,
    RerankError,
    TransientError,
)
from nanorag.generation.base import retry_after_seconds
from nanorag.ratelimit import Backoff, RateLimiter, call_with_retry
from nanorag.types import ScoredChunk

PROVIDER = "jina"
API_KEY_ENV = "JINA_API_KEY"
BASE_URL = "https://api.jina.ai/v1"
DEFAULT_MODEL = "jina-reranker-v2-base-multilingual"

#: The ``ScoredChunk.source`` a successful rerank stamps (plan.md §5).
SOURCE = "rerank:jina"


class JinaReranker:
    """Cross-encoder reranking against the Jina Reranker API.

    Parameters
    ----------
    model
        Jina reranker model name.
    api_key
        Explicit key. Defaults to the ``JINA_API_KEY`` environment variable.
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
        No API key was given or found in the environment.

    """

    def __init__(
        self,
        model: str = DEFAULT_MODEL,
        *,
        api_key: str | None = None,
        timeout: float = 10.0,
        connect_timeout: float = 5.0,
        backoff: Backoff | None = None,
        limiter: RateLimiter | None = None,
        sleep: Callable[[float], None] = time.sleep,
        rng: random.Random | None = None,
        transport: httpx.BaseTransport | None = None,
    ) -> None:
        """Resolve the key and open an ``httpx.Client`` (see the class docstring)."""
        if api_key is None:
            api_key = os.environ.get(API_KEY_ENV)
            if not api_key:
                raise ConfigError(
                    f"Jina reranking needs an API key: set {API_KEY_ENV}",
                    provider=PROVIDER,
                )
        self.model = model
        self._backoff = backoff if backoff is not None else Backoff()
        self._limiter = limiter
        self._sleep = sleep
        self._rng = rng
        self._client = httpx.Client(
            base_url=BASE_URL,
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
            },
            timeout=httpx.Timeout(timeout, connect=connect_timeout),
            transport=transport,
        )

    def __repr__(self) -> str:
        """Return the model only — never the key."""
        return f"JinaReranker(model={self.model!r})"

    def close(self) -> None:
        """Close the underlying HTTP client."""
        self._client.close()

    def rerank(
        self, query: str, hits: list[ScoredChunk], top_n: int
    ) -> list[ScoredChunk]:
        """Re-score *hits* against *query* and return the best ``top_n``.

        Raises
        ------
        RerankError
            The API call failed for any reason (auth, rate limit, transient
            network error, or an unexpected response shape).

        """
        if not hits:
            return []
        body = {
            "model": self.model,
            "query": query,
            "top_n": top_n,
            "documents": [hit.chunk.text for hit in hits],
        }
        try:
            results = call_with_retry(
                lambda: self._post(body),
                backoff=self._backoff,
                limiter=self._limiter,
                sleep=self._sleep,
                rng=self._rng,
            )
        except ProviderError as exc:
            raise RerankError(f"Jina reranking failed: {exc}") from exc
        # `_post` has already validated every `index` is in range for
        # `hits`, so this cannot raise IndexError.
        scored = [
            ScoredChunk(
                chunk=hits[r["index"]].chunk,
                score=r["relevance_score"],
                source=SOURCE,
            )
            for r in results
        ]
        # Jina documents `results` as already sorted best-first, but the
        # contract this method promises its caller (score-descending) is
        # cheap to guarantee outright rather than trust silently.
        scored.sort(key=lambda h: h.score, reverse=True)
        return scored[:top_n]

    def _post(self, body: dict[str, Any]) -> list[dict[str, Any]]:
        try:
            response = self._client.post("/rerank", json=body)
        except httpx.TimeoutException as exc:
            raise TransientError("request timed out", provider=PROVIDER) from exc
        except httpx.TransportError as exc:
            raise TransientError(
                f"connection failed: {exc}", provider=PROVIDER
            ) from exc
        if response.status_code != 200:
            raise self._error(response)
        n_documents = len(body["documents"])
        try:
            data = response.json()
            results = list(data["results"])
            for r in results:
                index = int(r["index"])
                float(r["relevance_score"])
                if not 0 <= index < n_documents:
                    raise ValueError(f"index {index} out of range")
        except (ValueError, KeyError, TypeError) as exc:
            raise ProviderError("unexpected response shape", provider=PROVIDER) from exc
        return results

    def _error(self, response: httpx.Response) -> ProviderError:
        """Map a non-200 response to a typed error (see the module docstring)."""
        status = response.status_code
        message = _error_message(response)
        context: dict[str, object] = {"provider": PROVIDER, "status": status}
        if status in (401, 403):
            return AuthError("Jina rejected the credentials", **context)
        if status == 429:
            return RateLimitError(
                "Jina rate limit hit",
                retry_after=retry_after_seconds(response.headers.get("Retry-After")),
                **context,
            )
        if status >= 500:
            return TransientError(f"Jina returned {status}", **context)
        return ProviderError(f"Jina returned {status}: {message}", **context)


def _error_message(response: httpx.Response) -> str:
    """Return a short, tolerant error message from a Jina error body."""
    try:
        data = response.json()
    except ValueError:
        return response.text[:200]
    detail = data.get("detail") if isinstance(data, dict) else None
    return str(detail) if detail else response.text[:200]
