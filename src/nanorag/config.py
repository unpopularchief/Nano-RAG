"""Runtime configuration: a single frozen ``Settings`` value.

There is no global settings singleton and no mutable module state — a
``Settings`` is built with :meth:`Settings.load` and passed explicitly.

Precedence, lowest to highest
-----------------------------
1. field defaults on this dataclass
2. ``[tool.nanorag]`` in the nearest ``pyproject.toml``
3. a named profile (``profile=`` argument or ``NANORAG_PROFILE`` env var)
4. individual ``NANORAG_<FIELD>`` environment variables
5. explicit keyword arguments to :meth:`Settings.load`

A profile is a committed bundle of field values for a common setup — ``local``
wires the fully-offline path. Individual env vars still override individual
profile values (step 4 over step 3), so ``NANORAG_PROFILE=local`` plus
``NANORAG_LOG_LEVEL=DEBUG`` works.
"""

from __future__ import annotations

import os
import tomllib
from dataclasses import dataclass, fields
from pathlib import Path
from typing import Any

from nanorag.errors import ConfigError

_LOG_LEVELS = {"CRITICAL", "ERROR", "WARNING", "INFO", "DEBUG", "NOTSET"}
_GENERATOR_PRESETS = {"auto", "groq", "gemini", "ollama"}

#: Named bundles of field values. Applied at precedence step 3.
_PROFILES: dict[str, dict[str, Any]] = {
    "local": {
        "generator_preset": "ollama",
        "embedding_model": "BAAI/bge-base-en-v1.5",
        "persist_dir": ".nanorag",
    },
}

_ENV_PREFIX = "NANORAG_"


@dataclass(frozen=True, slots=True)
class Settings:
    """Immutable configuration for one ``Rag`` instance.

    Attributes
    ----------
    persist_dir
        Directory for the SQLite database and vector-matrix snapshot.
    embedding_model
        ``fastembed`` model id the index is (or will be) built with.
    generator_preset
        ``"auto"`` (pick by available key/endpoint), or one of ``"groq"``,
        ``"gemini"``, ``"ollama"``.
    log_level
        Standard ``logging`` level name.
    request_timeout_s, connect_timeout_s
        Per-request and connection timeouts for provider HTTP calls.
    max_retries
        Retry budget for transient provider failures (429 / 5xx / timeout).
    retry_base_delay_s
        Base delay for exponential backoff with jitter.
    embed_batch_size
        Chunks per embedding batch.

    """

    persist_dir: str = ".nanorag"
    embedding_model: str = "BAAI/bge-base-en-v1.5"
    generator_preset: str = "auto"
    log_level: str = "WARNING"
    request_timeout_s: float = 30.0
    connect_timeout_s: float = 5.0
    max_retries: int = 3
    retry_base_delay_s: float = 0.5
    embed_batch_size: int = 64

    def __post_init__(self) -> None:
        """Validate every field; raise :class:`ConfigError` on a bad value."""
        if self.log_level not in _LOG_LEVELS:
            raise ConfigError(
                f"invalid log_level: {self.log_level!r}",
                allowed=sorted(_LOG_LEVELS),
            )
        if self.generator_preset not in _GENERATOR_PRESETS:
            raise ConfigError(
                f"invalid generator_preset: {self.generator_preset!r}",
                allowed=sorted(_GENERATOR_PRESETS),
            )
        for name in ("request_timeout_s", "connect_timeout_s", "retry_base_delay_s"):
            if getattr(self, name) <= 0:
                raise ConfigError(f"{name} must be > 0, got {getattr(self, name)!r}")
        if self.max_retries < 0:
            raise ConfigError(f"max_retries must be >= 0, got {self.max_retries!r}")
        if self.embed_batch_size < 1:
            raise ConfigError(
                f"embed_batch_size must be >= 1, got {self.embed_batch_size!r}"
            )
        if not self.persist_dir:
            raise ConfigError("persist_dir must be a non-empty string")
        if not self.embedding_model:
            raise ConfigError("embedding_model must be a non-empty string")

    @classmethod
    def load(
        cls,
        *,
        project_root: str | Path | None = None,
        profile: str | None = None,
        **overrides: Any,
    ) -> Settings:
        """Build a ``Settings`` by resolving every source in precedence order.

        Parameters
        ----------
        project_root
            Directory to look for ``pyproject.toml`` in. Defaults to the
            current working directory.
        profile
            Name of a profile in :data:`_PROFILES`. Falls back to the
            ``NANORAG_PROFILE`` environment variable.
        **overrides
            Explicit field values, highest precedence.

        Raises
        ------
        ConfigError
            An unknown profile, an unknown setting name, an unknown
            ``[tool.nanorag]`` key, or a malformed ``NANORAG_*`` value.

        """
        known = {f.name: f for f in fields(cls)}
        values: dict[str, Any] = {}

        root = Path(project_root) if project_root is not None else Path.cwd()
        values.update(_read_pyproject(root, set(known)))

        profile_name = (
            profile if profile is not None else os.environ.get("NANORAG_PROFILE")
        )
        if profile_name:
            if profile_name not in _PROFILES:
                raise ConfigError(
                    f"unknown profile: {profile_name!r}",
                    available=sorted(_PROFILES),
                )
            values.update(
                {k: v for k, v in _PROFILES[profile_name].items() if k in known}
            )

        for name, field_def in known.items():
            raw = os.environ.get(f"{_ENV_PREFIX}{name.upper()}")
            if raw is not None:
                values[name] = _coerce(raw, type(field_def.default), name)

        for key, value in overrides.items():
            if key not in known:
                raise ConfigError(f"unknown setting: {key!r}", known=sorted(known))
            values[key] = value

        return cls(**values)

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serialisable dict of every setting."""
        return {f.name: getattr(self, f.name) for f in fields(self)}


def _read_pyproject(root: Path, known: set[str]) -> dict[str, Any]:
    """Return the ``[tool.nanorag]`` table from ``root/pyproject.toml``, if any."""
    path = root / "pyproject.toml"
    if not path.is_file():
        return {}
    with path.open("rb") as handle:
        data = tomllib.load(handle)
    table = data.get("tool", {}).get("nanorag", {})
    if not isinstance(table, dict):
        raise ConfigError("[tool.nanorag] must be a table")
    unknown = set(table) - known
    if unknown:
        raise ConfigError(
            f"unknown [tool.nanorag] keys: {sorted(unknown)}", known=sorted(known)
        )
    return dict(table)


def _coerce(raw: str, target: type, field_name: str) -> Any:
    """Convert an environment-variable string to *target* (str / int / float)."""
    try:
        if target is int:
            return int(raw)
        if target is float:
            return float(raw)
        return raw
    except ValueError as exc:
        raise ConfigError(
            f"invalid value for {_ENV_PREFIX}{field_name.upper()}",
            value=raw,
            expected=target.__name__,
        ) from exc
