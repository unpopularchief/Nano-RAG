"""Deterministic fakes — the test infrastructure every later phase reuses.

Built in Phase A so the default suite needs no model download, no API key and
no real clock (plan.md §12).

- ``FakeEmbedder`` — a hashing vectoriser. Texts that share words get similar
  vectors, so retrieval relationships are controllable in a test.
- ``FakeGenerator`` — a real ``Generator``: returns scripted (or default)
  responses, raises scripted exceptions, and records the exact ``Prompt`` it
  was handed.
- ``FakeClock`` — advances only when told; ``sleep`` records the request
  instead of blocking.
"""

from __future__ import annotations

import hashlib
from collections.abc import Sequence

import numpy as np

from nanorag.generation.base import Generation
from nanorag.hashing import normalize_text
from nanorag.prompting.templates import Prompt
from nanorag.types import Usage


class FakeEmbedder:
    """Deterministic hashing embedder. No model, no network.

    Each token is hashed to a dimension; the resulting count vector is
    L2-normalised. Identical text -> identical vector; overlapping vocabulary
    -> positive cosine similarity.
    """

    def __init__(self, dim: int = 64, model_id: str = "fake-embedder") -> None:
        if dim < 1:
            raise ValueError("dim must be >= 1")
        self.dim = dim
        self.model_id = model_id

    def _vector(self, text: str) -> np.ndarray:
        vec = np.zeros(self.dim, dtype=np.float32)
        for token in normalize_text(text).lower().split():
            digest = hashlib.sha1(token.encode("utf-8")).digest()
            idx = int.from_bytes(digest[:4], "big") % self.dim
            vec[idx] += 1.0
        norm = float(np.linalg.norm(vec))
        if norm:
            vec /= norm
        return vec

    def embed(self, texts: Sequence[str]) -> np.ndarray:
        """Return an ``(len(texts), dim)`` float32 array of unit vectors."""
        if len(texts) == 0:
            return np.empty((0, self.dim), dtype=np.float32)
        return np.stack([self._vector(t) for t in texts])

    def embed_query(self, text: str) -> np.ndarray:
        """Return the single ``(dim,)`` float32 unit vector for *text*."""
        return self._vector(text)


class FakeGenerator:
    """A ``Generator`` that returns scripted responses and records every prompt.

    Satisfies :class:`nanorag.generation.Generator` (the protocol's third
    implementation, after the two HTTP clients). Scripted entries may be
    strings (returned as the completion text) or exceptions (raised on that
    call) — the latter is how fallback and retry paths are driven.

    Parameters
    ----------
    responses
        Entries consumed one per ``generate`` call in order. Once exhausted,
        ``default`` is returned for every further call.
    default
        The fallback response text.
    model, provider
        Recorded on the returned ``Usage``.
    context_window, max_output_tokens
        What the pipeline budgets against; small by default so tests can
        hit truncation cheaply.
    """

    def __init__(
        self,
        responses: Sequence[str | Exception] | None = None,
        *,
        default: str = "Fake answer grounded in the context. [1]",
        model: str = "fake-generator",
        provider: str = "fake",
        context_window: int = 4096,
        max_output_tokens: int = 256,
    ) -> None:
        self._responses = list(responses) if responses is not None else []
        self._default = default
        self.model = model
        self.provider = provider
        self.context_window = context_window
        self.max_output_tokens = max_output_tokens
        self.calls: list[Prompt] = []

    @property
    def last_prompt(self) -> str | None:
        """The most recent prompt (flattened to text), or ``None``."""
        return self.calls[-1].as_text() if self.calls else None

    def queue(self, *responses: str | Exception) -> None:
        """Append more scripted responses (or exceptions to raise)."""
        self._responses.extend(responses)

    def generate(self, prompt: Prompt) -> Generation:
        """Record *prompt*; return the next scripted response or raise it."""
        self.calls.append(prompt)
        item: str | Exception = self._default
        if self._responses:
            item = self._responses.pop(0)
        if isinstance(item, Exception):
            raise item
        text = prompt.as_text()
        prompt_tokens = (len(text) + 3) // 4
        completion_tokens = (len(item) + 3) // 4
        return Generation(
            text=item,
            usage=Usage(
                provider=self.provider,
                model=self.model,
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
                total_tokens=prompt_tokens + completion_tokens,
            ),
        )


class FakeClock:
    """A monotonic clock that only moves when told to.

    ``sleep`` records the requested duration and advances the clock by it,
    so backoff logic can be tested without wall-clock delay.
    """

    def __init__(self, start: float = 0.0) -> None:
        self._now = float(start)
        self.sleeps: list[float] = []

    def now(self) -> float:
        """Return the current time."""
        return self._now

    def monotonic(self) -> float:
        """Alias for :meth:`now` — matches ``time.monotonic``'s signature."""
        return self._now

    def sleep(self, seconds: float) -> None:
        """Record *seconds* and advance the clock by it."""
        if seconds < 0:
            raise ValueError("cannot sleep a negative duration")
        self.sleeps.append(seconds)
        self._now += seconds

    def advance(self, seconds: float) -> None:
        """Move the clock forward by *seconds* without recording a sleep."""
        if seconds < 0:
            raise ValueError("cannot move the clock backwards")
        self._now += seconds
