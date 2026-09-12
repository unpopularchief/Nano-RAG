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
- **Phase B1 — loaders, cleaning, ingest reporting.**
  - `types.py`: `LoadIssue` (`source_uri`, `reason`) and `IngestReport`
    (`loaded`, `skipped`, `failed` tuples) — frozen, validated, same family as
    `Document`/`Answer`. This is the B1-scoped shape; `added`/`updated`
    tracking waits for Phase E's `sync_path()` to have something to diff
    against.
  - `cleaning/normalize.py`: `clean_text` — whitespace/unicode/line-ending
    normalisation only (NFC + LF via the shared `hashing.normalize_text`,
    trailing-whitespace strip, blank-line collapsing). Not called
    automatically by any loader; boilerplate stripping is Phase G.
  - `loaders/base.py`: the `Loader` protocol and `read_text_file` — decodes
    `utf-8-sig` then falls back to `latin-1` (never fails on encoding alone),
    rejects files over `DEFAULT_MAX_FILE_SIZE` (20 MiB) before reading them
    into memory, and treats an embedded NUL byte as the binary-file signal.
  - `loaders/text.py`, `loaders/markdown.py`: `TextLoader`, `MarkdownLoader`.
  - `loaders/directory.py`: `DirectoryLoader.load_path()` — walks with
    `os.walk(followlinks=False)`, never reads or descends into a symlink
    (recorded in `skipped`), dispatches by extension, and always derives
    `source_uri` as `path.relative_to(root).as_posix()` so `doc_id` and
    `content_hash` are identical on Windows and Linux for the same corpus.
    One `LoaderError` from a file fails only that file.
  - `loaders/globbing.py`: `glob_match` — gitignore-style glob-to-regex
    translation (`*` stays within one path segment, `**` crosses segments),
    used for both `glob` and `ignore` matching. Deliberately not `fnmatch`,
    whose `*` crosses `/` and silently drops top-level files from the
    default `"**/*"` pattern.
  - Fixture corpus in `tests/data/corpus/` (UTF-8, UTF-8-BOM, latin-1, CRLF,
    empty, corrupt/NUL-byte, unrecognised extension, nested, ignored
    subdirectory) plus a hand-built 3-page `tests/data/sample.pdf`
    (committed now for Phase G, not parsed here).
  - 194 tests, 99% coverage on `src/` (the 1% gap is the symlink-skip
    branches, exercised only when the test environment can create symlinks —
    covered on CI, conditionally skipped locally without elevated
    permissions).
- **Phase B2 — chunking.**
  - `chunking/base.py`: the `Chunker` protocol (`chunk(document) -> list[Chunk]`)
    plus the splitting engine shared by all three chunkers — `validate_budget`
    (`ChunkingError` if `overlap_tokens` isn't strictly less than
    `target_tokens`), `make_chunk`, `extend_to_token_limit` (binary-searches
    the character offset fitting a token budget, for any `TokenCounter`,
    without assuming a fixed chars-per-token ratio), `atomic_spans` (the
    separator-hierarchy recursive splitter, falling back to character-level
    bisection once separators are exhausted), and `pack_spans` (greedily
    merges atomic spans into windows with proportional overlap).
  - `chunking/fixed.py`: `FixedChunker` — fixed-size character/token windows
    with no structural awareness, for when speed matters more than splitting
    on natural boundaries.
  - `chunking/recursive.py`: `RecursiveChunker` — the default (plan.md §7).
    Splits on a paragraph/line/sentence/word separator hierarchy, packed to
    a token budget.
  - `chunking/markdown.py`: `MarkdownChunker` — splits along ATX headings
    first (a chunk never straddles a heading boundary), then applies the
    same separator-hierarchy packing within each section. Every chunk
    carries the reserved `nanorag.heading_path` metadata key.
  - Every chunk is built so `document.text[start_char:end_char] ==
    chunk.text` and the union of every chunk's span covers the whole
    document with no gaps, regardless of overlap — verified by a Hypothesis
    property test across 500 generated `(text, target_tokens,
    overlap_tokens)` inputs run against all three chunkers (plan.md §9 B2
    checkpoint).
  - 238 tests, 100% coverage on the new `chunking/` package (99% on `src/`
    overall — the same pre-existing symlink-branch gap from B1).
- **Phase B3 — storage.**
  - `store/sqlite_docs.py`: `SqliteDocumentStore` — SQLite-backed
    `documents` / `chunks` / `embeddings` / `meta` tables.
    `upsert_document()` writes a document and replaces its chunks in one
    transaction; a re-ingest's stale chunks (and, via `ON DELETE CASCADE`,
    their embeddings) are removed, and a failure partway through leaves no
    row changed. `upsert_embeddings()` persists vectors keyed by chunk id
    under a recorded `(model_id, dim)`, raising `IndexModelMismatch` if a
    second embedding model or dimension is used against the same index.
    `iter_embeddings()` reads vectors back bit-exact, straight from the raw
    `float32` BLOB — how a `NumpyVectorStore` is rebuilt after a restart
    without re-running the embedder.
  - `store/numpy_store.py`: `NumpyVectorStore` — an in-memory exact-cosine
    index over L2-normalised vectors (`upsert`, `delete`, `compact`,
    `search`). Implements the concurrency discipline plan.md §5 locks in
    before Phase B: a single writer lock serialises mutations, and every
    mutation rebuilds the matrix / id list / id-map / alive-mask as new
    objects and rebinds all four together, so a `search` already holding a
    prior snapshot (taken under the lock, computed outside it) is never
    affected by a concurrent write. `delete()` tombstones rows; `compact()`
    physically reclaims them. `search()`'s optional `allowed_ids` filter is
    expressed in id space, not row-index space, so it survives a `compact()`.
  - 40 new tests (documents/chunks round-trip through SQLite including a
    Windows/Linux-stable schema, embeddings round-trip through BLOB
    bit-exact, a mid-batch chunk insert failure leaves zero rows for that
    document, a second embedding model/dim raises `IndexModelMismatch`, and
    a reopen-and-rebuild test reproduces `NumpyVectorStore.search()` results
    exactly — the B3 checkpoint). 100% coverage on the new `store/` package
    (99% on `src/` overall — the same pre-existing symlink-branch gap).
