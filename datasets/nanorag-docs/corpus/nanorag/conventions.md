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
  `import nanorag` never pulls `httpx`, `fastembed` or `onnxruntime`. `Rag`
  and the two HTTP clients resolve lazily (PEP 562 `__getattr__`) for the
  same reason; a test asserts the boundary in a clean subprocess.
- A client never lets a raw HTTP exception escape: every failure becomes one
  of the typed errors in `errors.py` (`AuthError`, `RateLimitError`,
  `QuotaExhausted`, `TransientError`, `ProviderError`), and the retry /
  fallback logic acts on the type, never on a status code. Retry only
  `RateLimitError` and `TransientError`; never a 4xx.
- The key is held privately and never appears in `repr()`, a log line or an
  exception. `__repr__` shows provider and model only.

## Facade

`Rag` composes public functions and owns no logic (plan.md §5 rule 1). Any
arithmetic the read path needs (`context_budget`, `insufficient_context`,
`load_vectors`) is a module-level function in `pipeline.py`, and a test
reproduces `Rag.query()` from those functions alone. If a behaviour exists
only inside the facade, that is a bug.

## Public types

`Answer`, `Citation`, `Usage` and `Timings` are frozen since `v0.1.0`
(Gate C, plan.md §18 F11): adding, removing or retyping a field is a
breaking change and bumps the minor while `0.x`. The three machine-readable
status signals — `Answer.insufficient_context`, `Answer.truncated`,
`Usage.provider` (who actually served, after fallback) — are part of that
contract: a caller never parses `text` to learn any of them.

## Abstention

`insufficient_context()` is a tunable heuristic, never a guarantee. Its
thresholds live on the embedder's score scale, so `Rag()` ships every
optional check off and only `from_defaults()` sets the floor measured for
the default embedder. Anything that changes the embedder re-measures
before it changes the default (the Gate C measurement is recorded in the
function docstring and README Limits).

## Configuration

One frozen `Settings` value, built by `Settings.load()` and passed
explicitly — no global singleton, no mutable module state. Precedence, low to
high: field defaults → `[tool.nanorag]` in `pyproject.toml` → named profile
(`profile=` / `NANORAG_PROFILE`) → individual `NANORAG_<FIELD>` env vars →
explicit keyword arguments. Any invalid value raises `ConfigError`.

## Command line

`cli/` is a thin shell over the facade: one module per subcommand, each
`run(args, settings) -> payload` + `render(payload) -> str`, so `--json`
and the text output are the same data and the JSON schema in `docs/cli.md`
is the whole contract. stdout is the result and nothing else; logging,
warnings and errors go to stderr. Exit codes map onto the error hierarchy
(`ConfigError` 3, `ProviderError` 4, `StoreError` 5, other `NanoRagError`
1, usage 2); an exception that is not a `NanoRagError` propagates. No
logic lives in the CLI that a library caller cannot reach.

## Keys & secrets

Environment variables only (`GROQ_API_KEY`, `GEMINI_API_KEY`, `JINA_API_KEY`).
Never written to disk, never logged, never in `repr()` or exception text. The
CLI's `.env` read exports into the environment where unset and prints
nothing. CI never holds a key.

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
