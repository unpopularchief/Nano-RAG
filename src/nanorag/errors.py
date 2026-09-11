"""The exception hierarchy every ``nanorag`` component raises.

One base class, ``NanoRagError``, so a caller can catch everything from this
library with a single ``except``. Each error carries an optional ``context``
dict for debugging; values whose key looks like a credential
(``api_key``, ``GROQ_API_KEY``, ``authorization``, …) are redacted from
``str()`` and ``repr()``. The library itself never places a secret in
``context`` — the redaction is a second line of defence for callers who do.

Hierarchy
---------
``NanoRagError``
    ``ConfigError``
    ``LoaderError``
    ``ChunkingError``
    ``EmbeddingError``
    ``StoreError``
        ``IndexModelMismatch``
    ``RetrievalError``
    ``GenerationError``
    ``ProviderError``
        ``AuthError``
        ``RateLimitError``   (carries ``retry_after``)
        ``TransientError``
        ``QuotaExhausted``

``ProviderError`` sits directly under ``NanoRagError`` rather than under
``GenerationError`` because hosted embedding backends (Phase G) raise it too.
"""

from __future__ import annotations

_SECRET_HINTS = (
    "key",
    "token",
    "secret",
    "password",
    "passwd",
    "authorization",
    "auth",
    "bearer",
    "credential",
    "cookie",
    "session",
)

_REDACTED = "***redacted***"


def _looks_secret(name: str) -> bool:
    """Return True if a context key name suggests a credential."""
    lowered = name.lower()
    return any(hint in lowered for hint in _SECRET_HINTS)


def _redact(context: dict[str, object]) -> dict[str, object]:
    """Copy *context*, masking values whose key looks like a credential."""
    return {
        key: (_REDACTED if _looks_secret(key) else value)
        for key, value in context.items()
    }


class NanoRagError(Exception):
    """Base class for every error raised by ``nanorag``.

    Parameters
    ----------
    message
        Human-readable description. Must not contain a secret.
    **context
        Arbitrary debugging fields. Values under credential-looking keys are
        redacted from ``str()`` / ``repr()``.

    """

    def __init__(self, message: str, **context: object) -> None:
        """Store *message* and copy *context* (see the class docstring)."""
        super().__init__(message)
        self.message = message
        self.context: dict[str, object] = dict(context)

    def _rendered_context(self) -> str:
        """Return the redacted context as `` (k=v, ...)``, or ``""`` if empty."""
        if not self.context:
            return ""
        pairs = ", ".join(f"{k}={v!r}" for k, v in _redact(self.context).items())
        return f" ({pairs})"

    def __str__(self) -> str:
        """Return the message followed by the redacted context."""
        return f"{self.message}{self._rendered_context()}"

    def __repr__(self) -> str:
        """Return ``ClassName('message' (k=v, ...))`` with context redacted."""
        return f"{type(self).__name__}({self.message!r}{self._rendered_context()})"


class ConfigError(NanoRagError):
    """Invalid configuration, a missing extra, or an unavailable component."""


class LoaderError(NanoRagError):
    """A source could not be read or decoded into a ``Document``."""


class ChunkingError(NanoRagError):
    """A document could not be split into chunks."""


class EmbeddingError(NanoRagError):
    """An embedder failed to produce vectors for a batch."""


class StoreError(NanoRagError):
    """The document store or vector store failed or is in an invalid state."""


class IndexModelMismatch(StoreError):
    """A request's embedding model or dimension does not match the index.

    The index records the ``model_id`` and ``dim`` it was built with; mixing a
    second embedding model into the same index is refused rather than silently
    corrupting similarity scores.
    """


class RetrievalError(NanoRagError):
    """Retrieval failed (bad filter grammar, empty index, dimension mismatch)."""


class GenerationError(NanoRagError):
    """Answer generation failed for a reason not attributable to a provider."""


class ProviderError(NanoRagError):
    """A remote provider returned an error or unusable response.

    Raised by both generation clients and hosted embedding backends. The
    subclasses below carry the distinctions the retry and fallback logic act
    on.
    """


class AuthError(ProviderError):
    """The provider rejected the credentials (HTTP 401/403). Never retried."""


class RateLimitError(ProviderError):
    """The provider rate-limited the request (HTTP 429).

    Parameters
    ----------
    message
        Description of the limit that was hit.
    retry_after
        Seconds to wait before retrying, from the provider's ``Retry-After``
        header when present. ``None`` if the provider did not say.
    **context
        Extra debugging fields.

    """

    def __init__(
        self,
        message: str,
        *,
        retry_after: float | None = None,
        **context: object,
    ) -> None:
        """Store *retry_after* and forward *message* / *context* to the base."""
        if retry_after is not None:
            context = {"retry_after": retry_after, **context}
        super().__init__(message, **context)
        self.retry_after = retry_after


class TransientError(ProviderError):
    """A transient failure (HTTP 5xx, timeout, connection reset). Retryable."""


class QuotaExhausted(ProviderError):
    """A hard quota (daily request cap, token budget) is spent.

    Triggers fallback to the alternate provider rather than a retry.
    """
