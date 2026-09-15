"""Named presets for the OpenAI-wire-format services (plan.md §4).

Groq, OpenRouter, Ollama, Together, LM Studio and vLLM all speak the same
chat-completions schema, so one client (:mod:`~nanorag.generation.openai_compat`)
parameterised by a ``Preset`` covers them all. A new service is a new
``Preset`` line, not a new file. Gemini has its own schema and its own
client.

``context_window`` per preset is the *default* the client assumes for the
preset's default model; pass ``context_window=`` to the client when using a
different model. Ollama's is deliberately small: Ollama serves every model
with a fixed context length (``OLLAMA_CONTEXT_LENGTH``, 4096 by default)
regardless of what the model could do, so a budget read from the model
card would silently overflow (plan.md §11).
"""

from __future__ import annotations

from dataclasses import dataclass

from nanorag.errors import ConfigError


@dataclass(frozen=True, slots=True)
class Preset:
    """One OpenAI-compatible service.

    Attributes
    ----------
    name
        Short provider name, recorded on ``Usage.provider``.
    base_url
        The ``/v1`` root the chat-completions path is appended to.
    api_key_env
        Environment variable holding the key, or ``None`` for a keyless
        local server.
    default_model
        Model used when the client is not given one.
    context_window
        Default window (tokens) assumed for ``default_model``.

    """

    name: str
    base_url: str
    api_key_env: str | None
    default_model: str
    context_window: int

    def __post_init__(self) -> None:
        """Validate the field values (see the class docstring)."""
        if not self.name or not self.base_url or not self.default_model:
            raise ConfigError("Preset name, base_url and default_model are required")
        if self.context_window < 1:
            raise ConfigError(f"context_window must be >= 1, got {self.context_window}")


#: ``llama-3.3-70b-versatile`` was gone from ``GET /v1/models`` entirely by
#: the Gate C check (2026-09-15, live key) — Groq retires names often, which
#: is why F6 exists. ``openai/gpt-oss-120b`` is Groq's current largest
#: general chat model (131 072 context, ``active: true``).
GROQ = Preset(
    name="groq",
    base_url="https://api.groq.com/openai/v1",
    api_key_env="GROQ_API_KEY",
    default_model="openai/gpt-oss-120b",
    context_window=131_072,
)

#: OpenRouter's free pool rotates: ``meta-llama/llama-3.3-70b-instruct:free``
#: was gone by the Gate C check (2026-09-15). A retired name surfaces as
#: ``ProviderError`` (``model_not_found``) and falls through the chain
#: (plan.md §18 F6); pass ``model=`` when this one goes too.
OPENROUTER = Preset(
    name="openrouter",
    base_url="https://openrouter.ai/api/v1",
    api_key_env="OPENROUTER_API_KEY",
    default_model="google/gemma-4-31b-it:free",
    context_window=262_144,
)

OLLAMA = Preset(
    name="ollama",
    base_url="http://localhost:11434/v1",
    api_key_env=None,
    default_model="qwen2.5:7b-instruct",
    context_window=4_096,
)

#: Every shipped preset, by name — the CLI's lookup table (Phase D).
PRESETS: dict[str, Preset] = {p.name: p for p in (GROQ, OPENROUTER, OLLAMA)}
