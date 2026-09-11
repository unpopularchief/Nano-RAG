# Changelog

All notable changes to this project are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and the project
follows [Semantic Versioning](https://semver.org/) from `v0.1.0` (while `0.x`,
breaking changes bump the minor).

## [Unreleased]

### Added

- **Phase A1 — scaffolding.** `git init`, MIT `LICENSE`, `.gitignore`,
  `.gitattributes` (fixture line endings and the vendored tiktoken cache are
  left byte-exact), `pyproject.toml` (hatchling, `src/` layout, three core
  dependencies, `local` / `pdf` / `html` / `dev` extras, ruff `E,F,I,UP,B,NPY,D`
  at 88 columns, mypy strict on `src/`, opt-in pytest markers).
- `.pre-commit-config.yaml` mirroring CI, `.github/workflows/ci.yml` (Ubuntu +
  Windows × Python 3.11 / 3.12 / 3.13), PR template, Dependabot for Actions.
- Doc skeletons: `README.md`, `CONTRIBUTING.md`, `ROADMAP.md`,
  `docs/conventions.md`, `docs/providers.md`.
- `src/nanorag/__init__.py` exporting `__version__` only, and a placeholder
  test suite.
- **Phase A2 — errors, hashing, core types.**
  - `errors.py`: the `NanoRagError` hierarchy (`ConfigError`, `LoaderError`,
    `ChunkingError`, `EmbeddingError`, `StoreError` → `IndexModelMismatch`,
    `RetrievalError`, `GenerationError`, and `ProviderError` →
    `AuthError` / `RateLimitError` (carries `retry_after`) / `TransientError` /
    `QuotaExhausted`). Each carries a `context` dict; credential-looking keys
    are redacted from `str()` and `repr()`.
  - `hashing.py`: `normalize_text` (NFC + LF — the one shared transform),
    `content_hash`, `stable_doc_id`, `chunk_id` (hashes the chunk's own
    normalised text). Deterministic across processes, platforms and
    `PYTHONHASHSEED`.
  - `types.py`: frozen, slotted, validated dataclasses — `Document`, `Chunk`,
    `ScoredChunk` (stable spine) plus `Citation`, `Usage`, `Timings`, `Answer`
    (explicitly unstable until Gate C, per plan.md §18 F11). Flat
    `str -> JSON scalar` metadata, copied into a read-only mapping on
    construction. `to_dict()` on every type for the future CLI.
  - Tests: hash determinism verified in subprocesses under two
    `PYTHONHASHSEED` values; type immutability, metadata read-only
    enforcement, field validation, equality and JSON round-trip.
- **Phase A3 — token counting, configuration, test fakes.**
  - `tokens.py`: `TokenCounter` protocol (one method, `count`);
    `TiktokenCounter` (lazy `tiktoken` import, encoder cached, honours an
    explicit `cache_dir` / `TIKTOKEN_CACHE_DIR`); `HeuristicCounter`
    (`ceil(len/4)`, warns on construction).
  - `config.py`: frozen `Settings` with `Settings.load()`. Precedence
    (low→high): field defaults, `[tool.nanorag]` in `pyproject.toml`
    (`tomllib`), a named profile (`profile=` / `NANORAG_PROFILE`), individual
    `NANORAG_<FIELD>` env vars, explicit kwargs. Ships the `local` profile
    (offline path). No global singleton or mutable module state. Every bad
    value raises `ConfigError`.
  - `tests/fakes.py`: `FakeEmbedder` (hashing vectoriser — shared vocabulary
    raises cosine similarity), `FakeGenerator` (scripted responses, records
    every prompt), `FakeClock` (advances only on `sleep`/`advance`).
  - `__init__.py` now re-exports the error hierarchy and the core types
    (`Document`, `Chunk`, `ScoredChunk`, `Citation`, `Usage`, `Timings`,
    `Answer`, `JsonScalar`). `import nanorag` still pulls in no `httpx`,
    `numpy`, `fastembed`, `onnxruntime` or `tiktoken` (asserted in a clean
    subprocess).
  - 147 tests, 100% coverage on `src/`.
- `[tool.mypy]` now targets `python_version = "3.12"` (numpy's stubs need it;
  the ≥3.11 runtime floor is enforced by ruff `UP` + `requires-python`).
