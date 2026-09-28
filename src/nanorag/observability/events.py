"""Small, content-free lifecycle events for one ``Rag`` instance."""

from __future__ import annotations

import json
import logging
from collections.abc import Callable
from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class Event:
    """A named event with numeric measurements, never document or prompt text."""

    name: str
    values: dict[str, int | float | bool]


EventHook = Callable[[Event], None]


def emit(
    hook: EventHook | None,
    name: str,
    **values: int | float | bool,
) -> None:
    """Log a JSON event and notify an optional hook without exposing content."""
    event = Event(name, values)
    logging.getLogger("nanorag").info(
        "%s", json.dumps({"event": name, **values}, sort_keys=True)
    )
    if hook is not None:
        try:
            hook(event)
        except Exception:
            logging.getLogger("nanorag").warning("event hook failed: %s", name)
