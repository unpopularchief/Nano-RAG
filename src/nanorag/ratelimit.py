"""Rate limiting and backoff around the one metered call per query.

Only generation is metered (plan.md §4): embeddings are local. So this is a
thin guard, not a framework — a token bucket per limit (requests/min,
tokens/min, requests/day), a ``Retry-After`` hold, and one retry loop with
bounded exponential backoff and full jitter. It knows nothing about HTTP or
providers (plan.md §6): callers hand it a callable that raises the typed
errors from :mod:`nanorag.errors`, and it decides only *whether* and *how
long* to wait.

Retry policy (plan.md §11 "Provider failures"): retry on ``RateLimitError``
and ``TransientError`` only — a 429, a 5xx, a timeout. Never on any other
4xx (``AuthError``, ``ProviderError``) and never on ``QuotaExhausted``; those
go straight to the caller, where the fallback chain decides.

Everything here is per-process (plan.md §18 F1): one process per key.
The clock and sleep are injectable so every test runs without waiting.
"""

from __future__ import annotations

import random
import time
from collections.abc import Callable
from dataclasses import dataclass
from typing import TypeVar

from nanorag.errors import ConfigError, RateLimitError, TransientError

T = TypeVar("T")

_MINUTE = 60.0
_DAY = 86_400.0


class _Bucket:
    """A token bucket: *capacity* units refilled evenly over *period* seconds."""

    def __init__(self, capacity: int, period: float, now: float) -> None:
        if capacity < 1:
            raise ConfigError(f"rate limit capacity must be >= 1, got {capacity}")
        self.capacity = capacity
        self.rate = capacity / period
        self.tokens = float(capacity)
        self.updated = now

    def refill(self, now: float) -> None:
        self.tokens = min(self.capacity, self.tokens + (now - self.updated) * self.rate)
        self.updated = now

    def wait_for(self, cost: int) -> float:
        """Seconds until *cost* tokens are available (0 if they are now)."""
        if cost <= self.tokens:
            return 0.0
        return (cost - self.tokens) / self.rate


class RateLimiter:
    """Token buckets for requests/min, tokens/min and requests/day.

    Parameters
    ----------
    rpm, tpm, rpd
        Per-minute request cap, per-minute token cap, per-day request cap.
        ``None`` disables that bucket. All three default to ``None`` — an
        unconfigured limiter never waits.
    clock, sleep
        Injectable ``time.monotonic`` / ``time.sleep`` for tests.

    Raises
    ------
    ConfigError
        A cap below 1.

    """

    def __init__(
        self,
        *,
        rpm: int | None = None,
        tpm: int | None = None,
        rpd: int | None = None,
        clock: Callable[[], float] = time.monotonic,
        sleep: Callable[[float], None] = time.sleep,
    ) -> None:
        """Create the configured buckets, all full."""
        self._clock = clock
        self._sleep = sleep
        now = clock()
        self._requests = _Bucket(rpm, _MINUTE, now) if rpm is not None else None
        self._tokens = _Bucket(tpm, _MINUTE, now) if tpm is not None else None
        self._daily = _Bucket(rpd, _DAY, now) if rpd is not None else None
        self._not_before = now

    def hold(self, seconds: float) -> None:
        """Make every ``acquire`` wait until *seconds* from now (``Retry-After``)."""
        if seconds > 0:
            self._not_before = max(self._not_before, self._clock() + seconds)

    def acquire(self, tokens: int = 0) -> float:
        """Wait until one request of *tokens* tokens is allowed, then spend it.

        Parameters
        ----------
        tokens
            Estimated prompt + completion tokens of the request, charged
            against the per-minute token bucket (if any).

        Returns
        -------
        float
            Seconds slept, ``0.0`` if the request was allowed immediately.

        Raises
        ------
        RateLimitError
            *tokens* exceeds the per-minute token capacity outright — no
            amount of waiting would ever admit the request.

        """
        if self._tokens is not None and tokens > self._tokens.capacity:
            raise RateLimitError(
                "request exceeds the per-minute token capacity",
                tokens=tokens,
                tpm=self._tokens.capacity,
            )
        now = self._clock()
        waits = [max(0.0, self._not_before - now)]
        for bucket, cost in (
            (self._requests, 1),
            (self._tokens, tokens),
            (self._daily, 1),
        ):
            if bucket is not None:
                bucket.refill(now)
                waits.append(bucket.wait_for(cost))
        wait = max(waits)
        if wait > 0:
            self._sleep(wait)
            now = self._clock()
        for bucket, cost in (
            (self._requests, 1),
            (self._tokens, tokens),
            (self._daily, 1),
        ):
            if bucket is not None:
                bucket.refill(now)
                bucket.tokens -= cost
        return wait


@dataclass(frozen=True, slots=True)
class Backoff:
    """Bounded exponential backoff with full jitter.

    Attributes
    ----------
    max_retries
        Retries after the first attempt; ``0`` means a single attempt.
    base_delay
        Delay ceiling for the first retry, doubling per attempt.
    max_delay
        Absolute ceiling on a computed delay. A provider's ``Retry-After``
        is honoured as given, even above this.

    """

    max_retries: int = 3
    base_delay: float = 0.5
    max_delay: float = 30.0

    def __post_init__(self) -> None:
        """Validate the field values (see the class docstring)."""
        if self.max_retries < 0:
            raise ConfigError(f"max_retries must be >= 0, got {self.max_retries}")
        if self.base_delay <= 0 or self.max_delay <= 0:
            raise ConfigError("base_delay and max_delay must be > 0")

    def delay(
        self,
        attempt: int,
        *,
        retry_after: float | None = None,
        rng: random.Random | None = None,
    ) -> float:
        """Seconds to wait before retry number *attempt* (0-based).

        ``retry_after``, when the provider gave one, wins outright — it is
        the authoritative figure. Otherwise the delay is drawn uniformly
        from ``[0, min(max_delay, base_delay * 2**attempt)]`` (full jitter,
        so retrying clients do not synchronise).
        """
        if retry_after is not None and retry_after >= 0:
            return retry_after
        ceiling = min(self.max_delay, self.base_delay * (2**attempt))
        return (rng or random).uniform(0.0, ceiling)


def call_with_retry(
    fn: Callable[[], T],
    *,
    backoff: Backoff,
    limiter: RateLimiter | None = None,
    tokens: int = 0,
    sleep: Callable[[float], None] = time.sleep,
    rng: random.Random | None = None,
) -> T:
    """Call *fn*, retrying on ``RateLimitError`` / ``TransientError`` only.

    Each attempt first passes through *limiter* (if given); a 429's
    ``retry_after`` is also pushed into the limiter as a hold, so a
    concurrent caller on the same limiter waits too. Any other exception
    propagates untouched on the first occurrence. After ``max_retries``
    retries the last retryable error is re-raised.
    """
    attempt = 0
    while True:
        if limiter is not None:
            limiter.acquire(tokens)
        try:
            return fn()
        except (RateLimitError, TransientError) as exc:
            if attempt >= backoff.max_retries:
                raise
            retry_after = exc.retry_after if isinstance(exc, RateLimitError) else None
            if limiter is not None and retry_after:
                limiter.hold(retry_after)
            sleep(backoff.delay(attempt, retry_after=retry_after, rng=rng))
            attempt += 1
