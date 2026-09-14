"""The ``Generator`` protocol — prompt -> text + usage, per plan.md §6.

A generator's job, precisely: take a :class:`~nanorag.prompting.Prompt`,
return the model's text plus a :class:`~nanorag.types.Usage`, raising the
typed errors from :mod:`nanorag.errors` (never a raw HTTP exception), with
retries and timeouts handled inside. Explicitly not its job: parsing
citations, or deciding what went into the prompt.

Two things every generator exposes so the pipeline never hard-codes them
(plan.md §11 "Context limits"): ``context_window`` — the model's window in
tokens — and ``max_output_tokens`` — what it reserves for the completion.
The context budget is derived from those two.

``OpenAICompatGenerator`` and ``GeminiGenerator`` are the two
implementations (so this protocol is written at the second, plan.md §15
#1); ``FakeGenerator`` in ``tests/fakes.py`` and ``FallbackGenerator`` also
satisfy it. This module imports no HTTP client, so ``nanorag.pipeline`` can
depend on it without pulling ``httpx``.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, Protocol, runtime_checkable

from nanorag.prompting.templates import Prompt
from nanorag.types import Usage


@dataclass(frozen=True, slots=True)
class Generation:
    """What a generator returns for one prompt.

    Attributes
    ----------
    text
        The model's completion.
    usage
        Token accounting and the provider/model that actually served it.

    """

    text: str
    usage: Usage

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serialisable dict of this generation."""
        return {"text": self.text, "usage": self.usage.to_dict()}


@runtime_checkable
class Generator(Protocol):
    """Something that turns a ``Prompt`` into a ``Generation``.

    Structural, not a base class — any object with these attributes and
    method satisfies it.

    Attributes
    ----------
    provider
        Short provider name (``"groq"``, ``"gemini"``, ``"ollama"``, …),
        recorded on ``Usage.provider``.
    model
        The model name sent to the provider.
    context_window
        The model's context window in tokens.
    max_output_tokens
        Tokens reserved for the completion; the request's output cap.

    """

    provider: str
    model: str
    context_window: int
    max_output_tokens: int

    def generate(self, prompt: Prompt) -> Generation:
        """Return the completion for *prompt* (see the module docstring)."""
        ...


# --- helpers shared by the HTTP clients ---------------------------------------

_DAILY_QUOTA_HINTS = re.compile(r"per day|daily|\bRPD\b|\bTPD\b|PerDay", re.IGNORECASE)


def looks_like_daily_quota(message: str) -> bool:
    """Return True if a 429 message describes a daily cap, not a burst limit.

    Groq and Gemini both answer a spent daily allowance with the same status
    as a per-minute limit (429 / ``RESOURCE_EXHAUSTED``); only the message
    tells them apart. A daily cap is ``QuotaExhausted`` — retrying is
    pointless, the fallback provider is the answer — while a per-minute
    limit is a ``RateLimitError`` worth waiting out.

    Only day-specific wording counts (``per day``, ``RPD``/``TPD``,
    ``PerDay`` in a Gemini quota id). The bare word "quota" does not:
    Gemini's generic per-minute message is "Resource has been exhausted
    (e.g. check quota)". Erring toward ``RateLimitError`` is the safe side —
    the fallback chain still takes over once the retries are spent.
    """
    return bool(_DAILY_QUOTA_HINTS.search(message))


def retry_after_seconds(value: str | None) -> float | None:
    """Parse a ``Retry-After`` header value given in seconds.

    Returns ``None`` when absent or not a plain number (the HTTP-date form
    is not used by the providers targeted here) — the caller then falls
    back to its own backoff schedule.
    """
    if value is None:
        return None
    try:
        seconds = float(value.strip())
    except ValueError:
        return None
    return seconds if seconds >= 0 else None
