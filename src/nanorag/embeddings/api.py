"""Hosted embedding clients for Gemini, Jina, and OpenAI-compatible APIs.

All clients implement :class:`~nanorag.embeddings.base.Embedder`, normalize
vectors at the boundary, and keep provider dependencies to the existing
``httpx`` runtime dependency. Hosted embeddings send input text to the named
provider; local ONNX remains the default.
"""

from __future__ import annotations

import os
import random
import time
from collections.abc import Callable, Sequence
from typing import Any

import httpx
import numpy as np

from nanorag.errors import (
    AuthError,
    ConfigError,
    ProviderError,
    RateLimitError,
    TransientError,
)
from nanorag.generation.base import retry_after_seconds
from nanorag.ratelimit import Backoff, RateLimiter, call_with_retry

GEMINI_MODEL = "gemini-embedding-2"
JINA_MODEL = "jina-embeddings-v3"
OPENAI_MODEL = "text-embedding-3-small"
_GEMINI_DIM = 768
_JINA_DIM = 1024
_OPENAI_DIM = 1536


class _HostedEmbedder:
    """Shared HTTP, retry, response validation, and normalization logic."""

    provider: str

    def __init__(
        self,
        *,
        provider: str,
        model: str,
        dim: int,
        base_url: str,
        headers: dict[str, str],
        timeout: float,
        connect_timeout: float,
        backoff: Backoff | None,
        limiter: RateLimiter | None,
        sleep: Callable[[float], None],
        rng: random.Random | None,
        transport: httpx.BaseTransport | None,
        model_identity: str | None = None,
    ) -> None:
        if dim < 1:
            raise ConfigError(f"dim must be >= 1, got {dim}")
        self.provider = provider
        self.model = model
        self.dim = dim
        self.model_id = f"{model_identity or provider}:{model}:{dim}"
        self._backoff = backoff if backoff is not None else Backoff()
        self._limiter = limiter
        self._sleep = sleep
        self._rng = rng
        self._client = httpx.Client(
            base_url=base_url,
            headers=headers,
            timeout=httpx.Timeout(timeout, connect=connect_timeout),
            transport=transport,
        )

    def __repr__(self) -> str:
        """Return provider/model metadata without credentials."""
        return f"{type(self).__name__}(model={self.model!r}, dim={self.dim})"

    def close(self) -> None:
        """Close the underlying HTTP client."""
        self._client.close()

    def _request_vectors(
        self, endpoint: str, body: dict[str, Any], *, expected_count: int
    ) -> np.ndarray:
        vectors = call_with_retry(
            lambda: self._post_vectors(endpoint, body),
            backoff=self._backoff,
            limiter=self._limiter,
            sleep=self._sleep,
            rng=self._rng,
        )
        if vectors.shape[0] != expected_count:
            raise ProviderError(
                "embedding response has an unexpected number of vectors",
                provider=self.provider,
                expected_count=expected_count,
                actual_count=vectors.shape[0],
            )
        return vectors

    def _post_vectors(self, endpoint: str, body: dict[str, Any]) -> np.ndarray:
        try:
            response = self._client.post(endpoint, json=body)
        except httpx.TimeoutException as exc:
            raise TransientError("request timed out", provider=self.provider) from exc
        except httpx.TransportError as exc:
            raise TransientError(
                f"connection failed: {exc}", provider=self.provider
            ) from exc
        if response.status_code != 200:
            raise self._error(response)
        try:
            rows = self._parse_rows(response.json())
            vectors = np.asarray(rows, dtype=np.float32)
        except (ValueError, KeyError, IndexError, TypeError) as exc:
            raise ProviderError(
                "unexpected embedding response shape", provider=self.provider
            ) from exc
        if vectors.ndim != 2 or vectors.shape[1] != self.dim:
            raise ProviderError(
                "embedding response has an unexpected dimension",
                provider=self.provider,
                expected_dim=self.dim,
                actual_shape=vectors.shape,
            )
        if not np.isfinite(vectors).all():
            raise ProviderError(
                "embedding response contains non-finite values", provider=self.provider
            )
        return _l2_normalize(vectors)

    def _parse_rows(self, data: Any) -> list[Any]:
        raise NotImplementedError

    def _error(self, response: httpx.Response) -> ProviderError:
        status = response.status_code
        context: dict[str, object] = {"provider": self.provider, "status": status}
        if status in (401, 403):
            return AuthError(f"{self.provider} rejected the credentials", **context)
        if status == 429:
            return RateLimitError(
                f"{self.provider} rate limit hit",
                retry_after=retry_after_seconds(response.headers.get("Retry-After")),
                **context,
            )
        if status >= 500:
            return TransientError(f"{self.provider} returned {status}", **context)
        return ProviderError(f"{self.provider} returned {status}", **context)


class GeminiEmbedder(_HostedEmbedder):
    """Text embeddings via the Gemini API.

    Uses retrieval-specific task types for corpus text and query text. The
    default model emits 768-dimensional vectors; set ``dim`` to another
    supported output size when constructing an index.
    """

    def __init__(
        self,
        model: str = GEMINI_MODEL,
        *,
        api_key: str | None = None,
        dim: int = _GEMINI_DIM,
        timeout: float = 30.0,
        connect_timeout: float = 5.0,
        backoff: Backoff | None = None,
        limiter: RateLimiter | None = None,
        sleep: Callable[[float], None] = time.sleep,
        rng: random.Random | None = None,
        transport: httpx.BaseTransport | None = None,
    ) -> None:
        """Configure Gemini credentials, model, width and HTTP behavior."""
        if api_key is None:
            api_key = os.environ.get("GEMINI_API_KEY")
            if not api_key:
                raise ConfigError("Gemini embeddings need GEMINI_API_KEY")
        super().__init__(
            provider="gemini",
            model=model,
            dim=dim,
            base_url="https://generativelanguage.googleapis.com/v1beta",
            headers={"x-goog-api-key": api_key},
            timeout=timeout,
            connect_timeout=connect_timeout,
            backoff=backoff,
            limiter=limiter,
            sleep=sleep,
            rng=rng,
            transport=transport,
        )

    def embed(self, texts: Sequence[str]) -> np.ndarray:
        """Embed corpus texts as retrieval documents."""
        return self._embed(texts, "RETRIEVAL_DOCUMENT")

    def embed_query(self, text: str) -> np.ndarray:
        """Embed one retrieval query."""
        return np.asarray(self._embed([text], "RETRIEVAL_QUERY")[0], dtype=np.float32)

    def _embed(self, texts: Sequence[str], task: str) -> np.ndarray:
        if not texts:
            return np.empty((0, self.dim), dtype=np.float32)
        body = {
            "requests": [
                {
                    "model": f"models/{self.model}",
                    "content": {"parts": [{"text": text}]},
                    "embedContentConfig": {
                        "taskType": task,
                        "outputDimensionality": self.dim,
                    },
                }
                for text in texts
            ]
        }
        return self._request_vectors(
            f"/models/{self.model}:batchEmbedContents", body, expected_count=len(texts)
        )

    def _parse_rows(self, data: Any) -> list[Any]:
        return [row["values"] for row in data["embeddings"]]


class JinaEmbedder(_HostedEmbedder):
    """Text embeddings via Jina's hosted embeddings API."""

    def __init__(
        self,
        model: str = JINA_MODEL,
        *,
        api_key: str | None = None,
        dim: int = _JINA_DIM,
        timeout: float = 30.0,
        connect_timeout: float = 5.0,
        backoff: Backoff | None = None,
        limiter: RateLimiter | None = None,
        sleep: Callable[[float], None] = time.sleep,
        rng: random.Random | None = None,
        transport: httpx.BaseTransport | None = None,
    ) -> None:
        """Configure Jina credentials, model, width and HTTP behavior."""
        if api_key is None:
            api_key = os.environ.get("JINA_API_KEY")
            if not api_key:
                raise ConfigError("Jina embeddings need JINA_API_KEY")
        super().__init__(
            provider="jina",
            model=model,
            dim=dim,
            base_url="https://api.jina.ai/v1",
            headers={"Authorization": f"Bearer {api_key}"},
            timeout=timeout,
            connect_timeout=connect_timeout,
            backoff=backoff,
            limiter=limiter,
            sleep=sleep,
            rng=rng,
            transport=transport,
        )

    def embed(self, texts: Sequence[str]) -> np.ndarray:
        """Embed corpus texts as retrieval passages."""
        return self._embed(texts, "retrieval.passage")

    def embed_query(self, text: str) -> np.ndarray:
        """Embed one retrieval query."""
        return np.asarray(self._embed([text], "retrieval.query")[0], dtype=np.float32)

    def _embed(self, texts: Sequence[str], task: str) -> np.ndarray:
        if not texts:
            return np.empty((0, self.dim), dtype=np.float32)
        return self._request_vectors(
            "/embeddings",
            {
                "model": self.model,
                "input": list(texts),
                "task": task,
                "dimensions": self.dim,
                "normalized": True,
                "embedding_type": "float",
            },
            expected_count=len(texts),
        )

    def _parse_rows(self, data: Any) -> list[Any]:
        rows = data["data"]
        return [row["embedding"] for row in sorted(rows, key=lambda row: row["index"])]


class OpenAICompatEmbedder(_HostedEmbedder):
    """Text embeddings against any OpenAI-compatible ``/embeddings`` API.

    ``base_url`` is the API root, such as ``https://api.openai.com/v1``;
    provider-specific prefixes and custom models can be supplied directly.
    """

    def __init__(
        self,
        model: str = OPENAI_MODEL,
        *,
        api_key: str | None = None,
        api_key_env: str = "OPENAI_API_KEY",
        base_url: str = "https://api.openai.com/v1",
        dim: int = _OPENAI_DIM,
        timeout: float = 30.0,
        connect_timeout: float = 5.0,
        backoff: Backoff | None = None,
        limiter: RateLimiter | None = None,
        sleep: Callable[[float], None] = time.sleep,
        rng: random.Random | None = None,
        transport: httpx.BaseTransport | None = None,
    ) -> None:
        """Configure endpoint, credentials, model, width and HTTP behavior."""
        if api_key is None:
            api_key = os.environ.get(api_key_env)
            if not api_key:
                raise ConfigError(f"OpenAI-compatible embeddings need {api_key_env}")
        base_url = base_url.rstrip("/")
        super().__init__(
            provider="openai-compatible",
            model=model,
            dim=dim,
            base_url=base_url,
            headers={"Authorization": f"Bearer {api_key}"},
            timeout=timeout,
            connect_timeout=connect_timeout,
            backoff=backoff,
            limiter=limiter,
            sleep=sleep,
            rng=rng,
            transport=transport,
            model_identity=base_url,
        )

    def embed(self, texts: Sequence[str]) -> np.ndarray:
        """Embed a batch of texts."""
        texts = list(texts)
        if not texts:
            return np.empty((0, self.dim), dtype=np.float32)
        return self._request_vectors(
            "/embeddings",
            {"model": self.model, "input": texts, "dimensions": self.dim},
            expected_count=len(texts),
        )

    def embed_query(self, text: str) -> np.ndarray:
        """Embed one query text."""
        return np.asarray(self.embed([text])[0], dtype=np.float32)

    def _parse_rows(self, data: Any) -> list[Any]:
        rows = data["data"]
        return [row["embedding"] for row in sorted(rows, key=lambda row: row["index"])]


def _l2_normalize(vectors: np.ndarray) -> np.ndarray:
    """Return float32 vectors normalized along their last axis."""
    norms = np.linalg.norm(vectors, axis=-1, keepdims=True)
    norms = np.where(norms == 0, 1.0, norms)
    return (vectors / norms).astype(np.float32)
