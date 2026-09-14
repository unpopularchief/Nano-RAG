"""``observability.timing``: per-stage milliseconds under an injected clock."""

import pytest

from nanorag.observability import STAGES, Timer
from nanorag.types import Timings
from tests.fakes import FakeClock


def test_stage_names_are_every_timings_field_but_total():
    assert STAGES == {"embed", "retrieve", "rerank", "context", "generate"}


def test_stages_accumulate_and_total_is_wall_clock_since_creation():
    clock = FakeClock()
    timer = Timer(clock=clock.now)
    with timer.stage("retrieve"):
        clock.advance(0.010)
    clock.advance(0.005)  # work between stages counts toward total only
    with timer.stage("generate"):
        clock.advance(0.100)
    with timer.stage("retrieve"):
        clock.advance(0.002)
    t = timer.timings()
    assert isinstance(t, Timings)
    assert t.retrieve_ms == pytest.approx(12.0)
    assert t.generate_ms == pytest.approx(100.0)
    assert t.embed_ms == t.rerank_ms == t.context_ms == 0.0
    assert t.total_ms == pytest.approx(117.0)


def test_stage_is_recorded_even_when_the_block_raises():
    clock = FakeClock()
    timer = Timer(clock=clock.now)
    with pytest.raises(RuntimeError), timer.stage("context"):
        clock.advance(0.003)
        raise RuntimeError("boom")
    assert timer.timings().context_ms == pytest.approx(3.0)


def test_unknown_stage_is_rejected():
    with pytest.raises(ValueError), Timer().stage("total"):
        pass


def test_real_clock_produces_non_negative_numbers():
    timer = Timer()
    with timer.stage("embed"):
        pass
    t = timer.timings()
    assert t.embed_ms >= 0.0 and t.total_ms >= t.embed_ms
