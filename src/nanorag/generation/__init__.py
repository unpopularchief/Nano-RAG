"""Generation: prompt -> text + usage, behind the ``Generator`` protocol.

Two HTTP clients cover five-plus services (plan.md §4):
``OpenAICompatGenerator`` (Groq, OpenRouter, Ollama, … via a ``Preset``)
and ``GeminiGenerator``. ``FallbackGenerator`` chains any of them.

The protocol, result type, presets and fallback chain import no HTTP
client and are exported eagerly. The two concrete clients pull ``httpx``,
so they are resolved lazily on first attribute access — ``from
nanorag.generation import OpenAICompatGenerator`` works, while
``nanorag.pipeline`` (which needs only the protocol) stays ``httpx``-free
until ``Rag.from_defaults()`` actually picks a provider (plan.md §7).
"""

from __future__ import annotations

from typing import Any

from nanorag.generation.base import (
    Generation,
    Generator,
    looks_like_daily_quota,
    retry_after_seconds,
)
from nanorag.generation.fallback import FallbackGenerator
from nanorag.generation.presets import GROQ, OLLAMA, OPENROUTER, PRESETS, Preset

__all__ = [
    "Generation",
    "Generator",
    "looks_like_daily_quota",
    "retry_after_seconds",
    "FallbackGenerator",
    "Preset",
    "PRESETS",
    "GROQ",
    "OPENROUTER",
    "OLLAMA",
    "OpenAICompatGenerator",
    "GeminiGenerator",
]

_LAZY = {
    "OpenAICompatGenerator": "nanorag.generation.openai_compat",
    "GeminiGenerator": "nanorag.generation.gemini",
}


def __getattr__(name: str) -> Any:
    """Import a concrete client on first access (see the module docstring)."""
    if name in _LAZY:
        import importlib

        return getattr(importlib.import_module(_LAZY[name]), name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
