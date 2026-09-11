# Conventions

The house style every module follows. See [`plan.md`](../plan.md) for the
reasoning. This reuses the sibling project (`ML From Scratch`) toolchain
wholesale, with a higher Python floor (`>=3.11`) and a 3-version CI matrix.

## Layout

- `src/` layout, package `nanorag`, built with hatchling.
- **One thing per file.** No `models.py` / `utils.py` grab-bags.
- **A package directory is created when its second implementation exists** —
  until then it is a single module. No empty `__init__.py` scaffolding.

## Line length & tooling

- 88 columns, enforced by `ruff format`.
- Lint rules: `E, F, I, UP, B, NPY, D` (ignore `D203`, `D213` — NumPy-style
  docstrings follow D212: summary on the line after the opening quotes).
- `tests/**` is exempt from `D` — tests document intent through their names.
- `mypy` strict on `src/`, relaxed on `tests/`. Every extension point is a
  `typing.Protocol`; without the checker a provider returning the wrong shape
  is caught only by a test needing a live key.

## Docstrings

NumPy style. Every public function/class documents Parameters, Returns (or
Attributes, for a class), and Raises where relevant.

## Core types

Frozen dataclasses (`frozen=True, slots=True`): `Document`, `Chunk`,
`ScoredChunk`, `Answer` (plus `Citation`, `Usage`, `Timings`). One `Chunk`
type — `ScoredChunk` wraps rather than subclasses. Validating `__post_init__`.
Metadata is a flat `str -> JSON scalar` map; nesting breaks filter compilation.
Reserved `nanorag.*` metadata key namespace.

## Determinism & hashing

- `content_hash` and `chunk_id` normalise text the same way (NFC + LF line
  endings) via one shared helper in `hashing.py`, then sha256-truncate.
- `chunk_id` hashes the chunk's **own** normalised text, not the document
  hash, so an edit rotates only the ids of chunks that actually changed.
- Hashes are deterministic across processes, platforms and `PYTHONHASHSEED`.

## Providers

- Two generation files (`openai_compat.py`, `gemini.py`) cover five-plus
  services. A provider is a ~150-line HTTP client behind a `Protocol`.
- Core logic never imports a provider. Provider/embedder classes are reached
  through `nanorag.generation` / `nanorag.embeddings`, never the top level, so
  `import nanorag` never pulls `httpx`, `fastembed` or `onnxruntime`.

## Configuration

One frozen `Settings` value, built by `Settings.load()` and passed
explicitly — no global singleton, no mutable module state. Precedence, low to
high: field defaults → `[tool.nanorag]` in `pyproject.toml` → named profile
(`profile=` / `NANORAG_PROFILE`) → individual `NANORAG_<FIELD>` env vars →
explicit keyword arguments. Any invalid value raises `ConfigError`.

## Keys & secrets

Environment variables only (`GROQ_API_KEY`, `GEMINI_API_KEY`, `JINA_API_KEY`).
Never written to disk, never logged, never in `repr()` or exception text. CI
never holds a key.

## Concurrency

Sync core is an invariant. Concurrency lives in one bounded-`ThreadPoolExecutor`
batching module. Single writer lock serialises `upsert` / `delete` / `compact`;
readers take a consistent `(matrix, row_map)` snapshot under the lock, compute
outside it. The per-document SQLite transaction commits **before** the
in-memory matrix is mutated. Single-process only through 1.0.

## Tests

Default `pytest` run is offline, deterministic, keyless and model-free
(< 10 s at Phase A, budgeted < 30 s through 1.0). Slow/networked work sits
behind the `local` / `provider` / `eval` / `integration` markers. Windows
tests (CRLF, non-UTF-8, path separators, SQLite handle cleanup) are mandatory.
