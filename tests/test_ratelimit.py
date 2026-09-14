"""``ratelimit``: token buckets driven by ``FakeClock`` (no real sleeping),
``Retry-After`` holds, bounded jittered backoff, and the retry loop's policy
(429 / 5xx / timeout retried; everything else raised at once)."""

import random

import pytest

from nanorag.errors import (
    AuthError,
    ConfigError,
    ProviderError,
    QuotaExhausted,
    RateLimitError,
    TransientError,
)
from nanorag.ratelimit import Backoff, RateLimiter, call_with_retry
from tests.fakes import FakeClock


def _limiter(**caps):
    clock = FakeClock()
    return RateLimiter(clock=clock.monotonic, sleep=clock.sleep, **caps), clock


# --- RateLimiter --------------------------------------------------------------


def test_unconfigured_limiter_never_waits():
    limiter, clock = _limiter()
    assert [limiter.acquire(10_000) for _ in range(50)] == [0.0] * 50
    assert clock.sleeps == []


@pytest.mark.parametrize("cap", ["rpm", "tpm", "rpd"])
def test_caps_below_one_are_config_errors(cap):
    with pytest.raises(ConfigError):
        RateLimiter(**{cap: 0})


def test_rpm_bucket_admits_a_burst_then_waits_for_refill():
    limiter, clock = _limiter(rpm=3)
    assert [limiter.acquire() for _ in range(3)] == [0.0, 0.0, 0.0]
    waited = limiter.acquire()  # 4th request: one token refills every 20s
    assert waited == pytest.approx(20.0)
    assert clock.sleeps == [pytest.approx(20.0)]
    assert clock.now() == pytest.approx(20.0)


def test_rpm_bucket_refills_with_elapsed_time_without_sleeping():
    limiter, clock = _limiter(rpm=60)
    for _ in range(60):
        limiter.acquire()
    clock.advance(30)  # half a minute -> 30 tokens back
    assert [limiter.acquire() for _ in range(30)] == [0.0] * 30
    assert limiter.acquire() > 0


def test_tpm_bucket_charges_tokens_and_rejects_impossible_requests():
    limiter, clock = _limiter(tpm=1000)
    assert limiter.acquire(600) == 0.0
    waited = limiter.acquire(600)  # 200 short; refill is 1000/60 per second
    assert waited == pytest.approx(200 / (1000 / 60))
    with pytest.raises(RateLimitError):
        limiter.acquire(1001)


def test_rpd_bucket_refills_over_a_day():
    limiter, clock = _limiter(rpd=2)
    limiter.acquire()
    limiter.acquire()
    waited = limiter.acquire()
    assert waited == pytest.approx(86_400 / 2)


def test_hold_delays_the_next_acquire_by_retry_after():
    limiter, clock = _limiter()
    limiter.hold(7.5)
    assert limiter.acquire() == pytest.approx(7.5)
    assert limiter.acquire() == 0.0  # the hold is consumed once its time passes


def test_hold_takes_the_later_of_two_holds_and_ignores_non_positive():
    limiter, clock = _limiter()
    limiter.hold(5)
    limiter.hold(2)
    limiter.hold(0)
    limiter.hold(-1)
    assert limiter.acquire() == pytest.approx(5.0)


def test_acquire_waits_for_the_slowest_constraint():
    limiter, clock = _limiter(rpm=1, tpm=1000)
    limiter.acquire(1000)  # exhausts both
    limiter.hold(3)
    waited = limiter.acquire(1)  # rpm needs 60s, tpm ~0.06s, hold 3s
    assert waited == pytest.approx(60.0)


# --- Backoff ------------------------------------------------------------------


def test_backoff_validation():
    with pytest.raises(ConfigError):
        Backoff(max_retries=-1)
    with pytest.raises(ConfigError):
        Backoff(base_delay=0)
    with pytest.raises(ConfigError):
        Backoff(max_delay=-1)


def test_backoff_delay_is_jittered_within_a_doubling_bounded_ceiling():
    backoff = Backoff(base_delay=1.0, max_delay=5.0)
    rng = random.Random(0)
    for attempt, ceiling in enumerate([1.0, 2.0, 4.0, 5.0, 5.0]):
        samples = [backoff.delay(attempt, rng=rng) for _ in range(200)]
        assert all(0.0 <= s <= ceiling for s in samples)
        assert max(samples) > ceiling * 0.8  # actually spans the range


def test_backoff_honours_retry_after_verbatim_even_above_the_ceiling():
    backoff = Backoff(base_delay=1.0, max_delay=5.0)
    assert backoff.delay(0, retry_after=42.0) == 42.0
    assert backoff.delay(3, retry_after=0.0) == 0.0
    # A negative Retry-After is nonsense; fall back to the schedule.
    assert 0.0 <= backoff.delay(0, retry_after=-1.0, rng=random.Random(1)) <= 1.0


# --- call_with_retry ----------------------------------------------------------


class Script:
    """A callable that raises the scripted exceptions, then returns."""

    def __init__(self, *failures, result="ok"):
        self.failures = list(failures)
        self.result = result
        self.calls = 0

    def __call__(self):
        self.calls += 1
        if self.failures:
            raise self.failures.pop(0)
        return self.result


def test_retries_rate_limit_and_transient_then_succeeds():
    clock = FakeClock()
    fn = Script(RateLimitError("429"), TransientError("503"))
    out = call_with_retry(
        fn,
        backoff=Backoff(max_retries=3, base_delay=1.0),
        sleep=clock.sleep,
        rng=random.Random(0),
    )
    assert out == "ok"
    assert fn.calls == 3
    assert len(clock.sleeps) == 2


def test_gives_up_after_max_retries_and_reraises_the_last_error():
    clock = FakeClock()
    fn = Script(*[TransientError(f"try {i}") for i in range(5)])
    with pytest.raises(TransientError, match="try 2"):
        call_with_retry(fn, backoff=Backoff(max_retries=2), sleep=clock.sleep)
    assert fn.calls == 3


def test_zero_retries_means_a_single_attempt():
    fn = Script(RateLimitError("429"))
    with pytest.raises(RateLimitError):
        call_with_retry(fn, backoff=Backoff(max_retries=0), sleep=lambda s: None)
    assert fn.calls == 1


@pytest.mark.parametrize(
    "exc",
    [AuthError("401"), QuotaExhausted("daily"), ProviderError("400"), ValueError("x")],
)
def test_non_retryable_errors_are_raised_on_the_first_attempt(exc):
    fn = Script(exc)
    with pytest.raises(type(exc)):
        call_with_retry(fn, backoff=Backoff(max_retries=5), sleep=lambda s: None)
    assert fn.calls == 1


def test_retry_after_is_slept_and_pushed_into_the_limiter_as_a_hold():
    clock = FakeClock()
    limiter = RateLimiter(clock=clock.monotonic, sleep=clock.sleep)
    fn = Script(RateLimitError("429", retry_after=12.0))
    call_with_retry(
        fn, backoff=Backoff(max_retries=1), limiter=limiter, sleep=clock.sleep
    )
    # The retry itself slept 12s; by then the hold has expired, so the
    # limiter's own acquire added nothing on top.
    assert clock.sleeps == [12.0]
    assert clock.now() == 12.0


def test_limiter_is_consulted_before_every_attempt():
    clock = FakeClock()
    limiter = RateLimiter(rpm=1, clock=clock.monotonic, sleep=clock.sleep)
    fn = Script(TransientError("503"))
    call_with_retry(
        fn,
        backoff=Backoff(max_retries=1, base_delay=0.001),
        limiter=limiter,
        sleep=clock.sleep,
        rng=random.Random(0),
    )
    # Second attempt needed a second rpm token: ~60s of limiter wait.
    assert clock.now() >= 60.0
