"""``Timer`` — per-stage wall-clock timing that ends up on ``Answer.timings``.

A ``Timer`` is created at the start of one query and each pipeline stage runs
inside ``with timer.stage("retrieve"):``. ``timings()`` then returns a
:class:`~nanorag.types.Timings` — stages that never ran stay ``0.0``, and
``total_ms`` is the elapsed time since the timer was created, not the sum of
the stages (there is always some work between them).

The clock is injectable (``time.perf_counter`` by default) so tests can
assert exact numbers without sleeping.
"""

from __future__ import annotations

import time
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from dataclasses import fields

from nanorag.types import Timings

#: Names ``stage()`` accepts — every ``Timings`` field except ``total_ms``.
STAGES: frozenset[str] = frozenset(
    f.name.removesuffix("_ms") for f in fields(Timings) if f.name != "total_ms"
)


class Timer:
    """Accumulates milliseconds per named stage (see the module docstring).

    Parameters
    ----------
    clock
        Returns seconds; only differences are used.

    """

    def __init__(self, clock: Callable[[], float] = time.perf_counter) -> None:
        """Start the total clock now."""
        self._clock = clock
        self._start = clock()
        self._ms: dict[str, float] = {}

    @contextmanager
    def stage(self, name: str) -> Iterator[None]:
        """Time the enclosed block under *name*, adding to any earlier run.

        Raises
        ------
        ValueError
            *name* is not a ``Timings`` stage.

        """
        if name not in STAGES:
            raise ValueError(
                f"unknown stage {name!r}; expected one of {sorted(STAGES)}"
            )
        started = self._clock()
        try:
            yield
        finally:
            self._ms[name] = self._ms.get(name, 0.0) + (self._clock() - started) * 1e3

    def timings(self) -> Timings:
        """Return the stages recorded so far plus the total elapsed time."""
        return Timings(
            **{f"{name}_ms": ms for name, ms in self._ms.items()},
            total_ms=(self._clock() - self._start) * 1e3,
        )
