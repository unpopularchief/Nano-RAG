# Changelog

All notable changes to this project are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and the project
follows [Semantic Versioning](https://semver.org/) from `v0.1.0` (while `0.x`,
breaking changes bump the minor).

## [Unreleased]

### Added

- `citations/parser.py` (Phase D, session D1): `parse_citations` resolves a
  generated answer's `[n]` markers against the `Context` it was prompted
  with, dropping unresolvable markers while still counting them towards a
  reported validity rate. `Rag.query` now populates `Answer.citations` from
  this instead of always returning `()`; a marker that doesn't resolve is
  logged, not silently dropped.
- `cli/` (Phase D, session D2): the `nanorag` command — `ingest ROOT`
  (offline), `query QUESTION` (one generation call) and `inspect` (opens
  the store directly, loads no model), each with `--json` for a documented
  payload (`docs/cli.md`) and stable exit codes (`2` usage, `3`
  `ConfigError`, `4` `ProviderError`, `5` `StoreError`). stdout holds only
  the result; logs and errors go to stderr. `.env` in the working directory
  (or `--env-file`) is read for keys, never written or printed. Installed
  as a console script (`[project.scripts]`) and runnable as
  `python -m nanorag.cli`.
- `generation.NullGenerator`: the null object for a pipeline that only
  ingests — satisfies the `Generator` protocol, raises `ConfigError` on the
  first `generate`. `Rag.from_defaults(generator=NullGenerator())` builds
  an index without reading a key or probing Ollama.
- `SqliteDocumentStore.count_documents()` / `count_chunks(doc_id=None)`.

### Fixed

- `citations/parser.py` now reads fullwidth `【n】` markers as `[n]`.
  `openai/gpt-oss-120b` — the Groq preset default — writes them that way
  even when the prompt shows `[n]`; before this, every citation in a
  correct answer from the primary provider went unresolved *and* uncounted
  (a validity rate of "no markers", not "0 %"), so nothing was logged.

## [0.1.0] — 2026-09-15

The MVP: documents in, cited answer out, on a free key or fully offline.
First public release; everything below was built towards it.

### Gate C — decisions locked against the working read path

- **`Answer`, `Citation`, `Usage`, `Timings` are frozen** (plan.md §18
  F11). Fields as shipped in C3, unchanged: `Answer(text, citations,
  contexts, insufficient_context, truncated, usage, timings)`. A field
  change is now a breaking change.
- **`from_defaults()` auto-chains every available generator** — confirmed
  as implemented (Groq → Gemini → Ollama). Fallback is opt-out by naming a
  `generator_preset`, not opt-in.
- **Abstention heuristic locked by measurement** (plan.md §18 F10).
  Measured with the real `bge-base-en-v1.5` on two ~120-chunk corpora
  (the project's own docs; nine unrelated package READMEs), 33 answerable
  vs 32 unrelated questions, `k=8`. The relative top-1-vs-rest gap does not
  separate (answerable median 0.05–0.07 with a 0.005 floor; unrelated
  median 0.015, max 0.04) — `min_gap` stays off by default. The absolute
  top-1 score does (answerable ≥ 0.61, unrelated ≤ 0.60): new
  `insufficient_context(min_score=)` / `Rag(min_score=)` floor, off in
  `Rag()` because the scale is the embedder's, set to
  `DEFAULT_MIN_SCORE = 0.55` by `from_defaults()` for the default embedder
  only. Documented as tunable, not a guarantee.
- **F1–F4 confirmed in place**: single-process scope stated (README
  Limits, `docs/conventions.md`); writer lock + snapshot reads + SQLite-
  before-matrix ordering in `store/numpy_store.py` and `pipeline.py`; the
  ≥ 200-item eval set is a Phase D deliverable; token counting takes the
  15 % margin + `usage` reconciliation route with no `tokenizers`
  dependency.
- **Preset defaults verified against real keys, not just the models
  list** (a listed model can still 404 on generation): Groq's
  `llama-3.3-70b-versatile` was gone from `GET /v1/models` entirely →
  `openai/gpt-oss-120b`; Gemini's `gemini-2.5-flash` still listed but
  404'd on `generateContent` ("no longer available to new users") →
  `gemini-3.6-flash` (the model Google's own error names as the
  replacement); OpenRouter's `meta-llama/llama-3.3-70b-instruct:free` gone
  from its catalogue → `google/gemma-4-31b-it:free` (262k window). All
  three free tiers rotate names — exactly what F6's fallback exists for.
- Version `0.1.0`; classifier `Development Status :: 3 - Alpha`.

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
    (unstable until Gate C, per plan.md §18 F11 — frozen at `0.1.0`). Flat
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
- **Phase B4 — embeddings. Phase B is now feature-complete.**
  - `embeddings/base.py`: the `Embedder` protocol (`embed`, `embed_query`,
    `dim`, `model_id`). `FakeEmbedder` (Phase A) is its second
    implementation, so writing the protocol here does not break the
    "a Protocol is written at the second implementation" rule.
  - `embeddings/local.py`: `FastEmbedEmbedder` — local ONNX embeddings via
    `fastembed`, no torch. Lazy-imports `fastembed`/`onnxruntime` inside
    `__init__` (never at import time), raising `ConfigError` naming the exact
    `uv add "nanorag[local]"` install command if the extra is missing.
    Resolves a model's dimension via `TextEmbedding.get_embedding_size()`
    before constructing the ONNX session. Defaults to `threads=1` for
    deterministic execution (plan.md §18 F5) and explicitly L2-normalises
    every output vector at the boundary regardless of what the backend
    already does. Wraps third-party exceptions in `EmbeddingError`.
  - `embeddings/cache.py`: `EmbeddingCache` (SQLite-backed
    `(model_id, text_hash) -> vector` store) and `CachingEmbedder` (wraps any
    `Embedder`, serving `embed()` from the cache and computing only the
    misses). Keyed by `sha256(normalize_text(chunk text))`, never by
    `chunk_id` — a chunk whose text is unchanged still hits the cache even
    when an upstream edit shifted its ordinal (and therefore its id).
    `embed_query()` always delegates, uncached — queries are called once
    each and, for some models, are not the same vector space as a document
    embedding of the same text.
  - `embeddings/batching.py`: `batched()` and `BatchingEmbedder`, bounding
    how many texts reach an embedder in one call, independent of whatever
    batching the wrapped embedder already does. The one place any
    concurrency for embedding would live (plan.md §15 #21); stays
    synchronous — local embedding is CPU-bound with no rate limit to
    respect, so there is nothing yet for a thread pool to buy.
  - `.github/workflows/local.yml`: runs the opt-in `-m local` suite on one
    fixed runner image (plan.md §12), caching the downloaded ONNX model
    between runs.
  - 34 new tests: default-suite unit tests for the protocol, cache and
    batching against `FakeEmbedder` (no model download), plus an opt-in
    `-m local` suite against the real ONNX model — construction, L2
    normalisation, determinism within a build, the missing-`[local]`-extra
    `ConfigError`, and the B4 checkpoint itself: 10,000 chunks embedded
    locally, then a second run against a fresh `EmbeddingCache` at the same
    path makes zero calls to the model. 100% coverage on the new
    `embeddings/` package when the default and `-m local` suites are
    combined (99% on `src/` overall in the default run alone, since
    `local.py`'s model-calling lines are only exercised under `-m local` —
    same pre-existing symlink-branch gap otherwise).
- **Gate B review — closed.** Auditing plan.md §9's Phase B block against
  the actual code/tests (not just the session summaries) found the
  phase-level **Acceptance** bullet had never been verified end-to-end: B1-B4
  each tested their own component in isolation, but nothing wired the full
  pipeline together, and no test had ever actually blocked sockets to prove
  the "no network access" claim (it was only true by construction).
  - `tests/conftest.py`: `blocked_sockets` — a context manager that patches
    `socket.socket.connect`/`connect_ex` (not the class itself, since
    libraries construct real socket objects even on a call that never
    reaches the network) to raise, proving an offline claim rather than
    merely asserting no networking library was imported.
  - `tests/test_gate_b_offline_ingest.py` (new, `-m local`): the literal
    Phase B acceptance test. A synthetic 200-document corpus (10,680 chunks)
    goes through `DirectoryLoader` -> `FixedChunker` ->
    `CachingEmbedder(BatchingEmbedder(FastEmbedEmbedder(...)))` ->
    `SqliteDocumentStore` + `NumpyVectorStore`, entirely inside
    `blocked_sockets()`. The store is then closed, reopened under a second
    `blocked_sockets()` block, and the vector index rebuilt from SQLite
    reproduces the pre-close search results exactly; a second identical
    embed call is then proven to make zero calls to the real model.
  - `tests/test_loaders.py`: `test_text_loader_handles_a_5mb_file` — the 5 MB
    fixture named in the Phase B Tests bullet's fixture list, generated at
    test time into `tmp_path` rather than committed (B1 had deferred this;
    it was never actually added).
  - 324 tests passing across the combined default + `-m local` suite (2
    skipped, the usual symlink cause), 99% coverage on `src/`.
- **Phase C1 — dense retrieval and metadata pre-filtering.**
  - `store/filters.py`: the filter grammar and `compile_filter`. A flat,
    Mongo-shaped mapping — `key: scalar` shorthand, `$eq` / `$ne` / `$gt` /
    `$gte` / `$lt` / `$lte` / `$in` / `$nin` / `$prefix` / `$exists`, and
    `$and` / `$or` / `$not` — compiled to a SQL predicate over the
    `chunks` × `documents` join with every operand bound as a parameter
    (column names come from a fixed whitelist, never from the filter).
    `chunk_id` / `doc_id` / `ordinal` / `source_uri` address columns;
    any other key is a metadata key resolved chunk-first, then document,
    via `json_extract`. Every leaf evaluates to exactly 0/1 so `$not` is a
    plain negation; an absent key satisfies no comparison (use `$exists`);
    range operators compare numbers with numbers and text with text only
    (a `typeof` guard, since SQLite orders every number below every
    string); column operands are type-checked at compile time so column
    affinity can never silently coerce `ordinal = '3'`.
  - `SqliteDocumentStore.filter_chunk_ids(filter)` (the pre-filter: a set
    of chunk ids for `NumpyVectorStore.search(allowed_ids=...)`; matching
    nothing returns an empty set, not an error) and `get_chunks_by_ids()`
    (batched `IN` lookups; how hits become `Chunk`s again).
  - `NumpyVectorStore.search`: ties are now broken by chunk id rather than
    row position — row order is insertion order in a live index but
    `chunk_id` order after a rebuild from SQLite, so the old order could
    differ across a reopen. Implemented as partition-to-the-k-th-score +
    lexsort over the contenders, so a boundary tie is cut by id too. Also
    stops fancy-indexing a copy of the whole matrix on every search
    (that copy, not the dot product, was ~95% of the query time): rows are
    gathered only when candidates are under 25% of the index, otherwise
    the matrix is scored in place. 100k × 768 unfiltered search: ~5 ms
    p95 locally, from ~175 ms.
  - `retrieval/dense.py`: `DenseRetriever(embedder, vectors, docs, k=10)`
    with `retrieve(query, k=None, filter=None) -> list[ScoredChunk]`
    (`source="dense"`). Four readable steps: embed the query, resolve the
    filter in SQLite, search over the allowed ids, hydrate. Refuses a
    dimension mismatch at construction; a hit whose chunk is missing from
    the document store raises `RetrievalError` rather than being dropped
    silently. No `Retriever` protocol yet — it arrives with the second
    implementation (BM25, Phase F).
  - Tests (`tests/test_filters.py`, `tests/test_dense_retriever.py`, plus
    additions to `tests/test_numpy_store.py`): every operator against a
    real store; 26 malformed filters each raise `RetrievalError`; SQL
    injection via an operand is inert; a Hypothesis property test
    (300 examples, random metadata × random filters up to 4 leaves deep)
    asserts the compiled SQL agrees with a plain-Python reference
    evaluator; the C1 checkpoint — retrieval matches a brute-force NumPy
    reference exactly, unfiltered and filtered; a filter matching nothing
    returns `[]`; ties are identical before and after a reopen; an explicit
    test shows post-filtering the top-*k* loses every matching result where
    pre-filtering returns all of them (plan.md §15 #7).
  - 389 tests in the default suite, 100% coverage on `retrieval/`,
    `store/filters.py`, `store/numpy_store.py`, `store/sqlite_docs.py`
    (98% on `src/` overall: the symlink branches plus `local.py`'s
    model-calling lines, both covered only outside the default run).
- **Phase C2 — context building and prompting.**
  - `context/builder.py`: `ContextBuilder(budget_tokens, counter=)` with
    `build(hits) -> Context`. Input order is rank order and is preserved
    (the retriever, and later the reranker or fusion stage, owns ranking).
    Deduplicates by chunk id and by normalised text (NFC + LF, surrounding
    whitespace ignored) keeping the first occurrence, skips empty or
    whitespace-only chunks, then takes blocks in rank order until the first
    that does not fit — the context is always a rank-order prefix, never a
    top-*k* with holes, and a chunk is never split (its offsets are what
    citations point at). Blocks render as `[n]` on its own line followed by
    the chunk text verbatim, joined by blank lines. The budget check is on
    that rendered, joined text under the counter given — not on a sum of
    per-block counts — so it holds for tokenizers that merge across block
    boundaries. `Context` carries `blocks`, the read-only `chunk_id ->
    label` map, `text`, `token_count`, `budget_tokens` and `truncated`
    (never silent, plan.md §15 #6), and enforces `token_count <=
    budget_tokens` on construction. A budget below 1 is a `ConfigError`.
  - `prompting/fencing.py`: `make_nonce(avoid=)` — a `secrets`-random
    128-bit hex nonce redrawn until it does not occur in the text it will
    surround; `fence(body, nonce)` — wraps the body verbatim between
    `=== BEGIN UNTRUSTED CONTEXT <nonce> ===` and the matching `END` line
    and refuses a body containing its own nonce, so a closing fence cannot
    be forged from inside.
  - `prompting/templates.py`: `PromptBuilder(system_prompt=)` with
    `build(question, context) -> Prompt(system, user, nonce)`. The default
    system prompt is an instruction hierarchy: everything inside the fence
    is data, never instructions; answer only from it; cite with `[n]`, one
    label per bracket; abstain with exactly `INSUFFICIENT_CONTEXT_TEXT` (the
    defined "insufficient context" answer the pipeline will also return
    without a model call when retrieval yields nothing). The template must
    carry the `{nonce}` placeholder (`ConfigError` otherwise) because a
    model that is not told the nonce cannot tell a real fence from a forged
    one; the nonce is drawn per request and avoids both the context and the
    question. `Prompt.as_text()` flattens to one string for single-string
    generators.
  - Tests (`tests/test_context_builder.py`, `tests/test_prompting.py`):
    the C2 checkpoint — a 500-example Hypothesis test over adversarial
    chunk sizes (empty, whitespace-only, single characters, far over
    budget, re-inserted duplicates and same-text copies) and four counters
    (heuristic, word, super-additive separator-heavy, sub-additive
    distinct-characters) asserts the context never exceeds the budget, is a
    rank-order prefix of the deduplicated input, labels 1..n consistently,
    reports `truncated` exactly when something was dropped, and that the
    next candidate genuinely would not have fit; an explicit test shows a
    per-block sum would have exceeded the budget where the joined count
    does not; a real `TiktokenCounter` case; every dedup and truncation
    rule; the fence's forgery-resistance with a document that contains
    fake `END`/`BEGIN` lines; nonce redraw on collision; and an end-to-end
    context -> prompt -> `FakeGenerator` -> `[n]` resolution to a chunk id.
  - 448 tests in the default suite, 100% coverage on `context/` and
    `prompting/` (98% on `src/` overall, same pre-existing gaps).
- **Phase C3 — generation, rate limiting, the pipeline. Phase C is now
  feature-complete.**
  - `ratelimit.py`: `RateLimiter(rpm=, tpm=, rpd=)` — one token bucket per
    cap, refilled evenly, plus `hold(seconds)` for a provider's
    `Retry-After`; `acquire(tokens)` sleeps (injectable) until admitted and
    refuses outright a request larger than the per-minute token capacity.
    `Backoff(max_retries, base_delay, max_delay)` — exponential with full
    jitter, `Retry-After` honoured verbatim even above the ceiling.
    `call_with_retry(fn, backoff=, limiter=)` — retries `RateLimitError` and
    `TransientError` only (429, 5xx, timeouts); `AuthError`, `QuotaExhausted`
    and any other error propagate on the first occurrence (plan.md §11).
  - `generation/base.py`: the `Generator` protocol — `generate(Prompt) ->
    Generation(text, usage)` plus `provider`, `model`, `context_window` and
    `max_output_tokens`, so the context budget is read from the generator
    and never hard-coded (plan.md §11). Written at the second
    implementation, per the rule. Shared helpers `looks_like_daily_quota`
    (day-specific wording only — the bare word "quota" appears in Gemini's
    per-minute message) and `retry_after_seconds`.
  - `generation/presets.py`: `Preset(name, base_url, api_key_env,
    default_model, context_window)`; `GROQ` (`llama-3.3-70b-versatile`,
    131k), `OPENROUTER`, `OLLAMA` (`qwen2.5:7b-instruct`, **4096** — Ollama's
    server-side default context, not the model's).
  - `generation/openai_compat.py`: `OpenAICompatGenerator(preset, model,
    api_key=, context_window=, max_output_tokens=, temperature=, timeout=,
    backoff=, limiter=, transport=)` — `POST /chat/completions` with a
    system + user message. Error mapping: 401/403 → `AuthError`; 429 →
    `QuotaExhausted` if the body describes a daily cap (Groq's RPD/TPD),
    else `RateLimitError` carrying `Retry-After`; 404 / `model_not_found` →
    `ProviderError` with `code="model_not_found"` (plan.md §18 F6); 5xx,
    timeouts and connection failures → `TransientError`; other 4xx →
    `ProviderError`. The key is held privately and never appears in
    `repr()` or an exception.
  - `generation/gemini.py`: `GeminiGenerator` — `POST
    /models/{model}:generateContent` with `systemInstruction` + one user
    content; default `gemini-2.5-flash`. Same error mapping keyed on
    Gemini's `error.status` (`UNAUTHENTICATED` / `PERMISSION_DENIED`,
    `RESOURCE_EXHAUSTED`, `NOT_FOUND`).
  - `generation/fallback.py`: `FallbackGenerator(primary, *fallbacks)` —
    itself a `Generator`. Hands off to the next member on `QuotaExhausted`,
    any `ProviderError` (a retired model name included), and a
    `RateLimitError` / `TransientError` that survived the client's own
    retries; **not** on `AuthError` (a rejected key is surfaced, not papered
    over). Every hand-off is logged at `WARNING`; `Usage.provider` names
    the member that served; the chain's `context_window` /
    `max_output_tokens` are the minimum across members.
  - `generation/__init__.py`: protocol, presets and the chain are exported
    eagerly; the two `httpx` clients resolve lazily, so `nanorag.pipeline`
    stays `httpx`-free until `from_defaults()` picks a provider.
  - `observability/timing.py`: `Timer` — `with timer.stage("retrieve"):`
    accumulates milliseconds per `Timings` field; `timings()` fills
    `Answer.timings`, `total_ms` being wall clock since creation.
  - `pipeline.py`: `Rag(embedder=, generator=, docs=, vectors=None,
    chunker=, loader=, prompt_builder=, counter=, k=8, budget_margin=0.15,
    min_results=1, min_gap=0.0)`. `ingest_path()` / `ingest()` — chunk,
    embed, `upsert_document` + `upsert_embeddings` in SQLite **then** update
    the in-memory index (stale rows of a re-ingested document deleted from
    it; plan.md §18 F2 ordering). `retrieve()` — the retriever. `query()` —
    retrieve → `context_budget()` → `ContextBuilder` → `insufficient_context()`
    → `PromptBuilder` → `generator.generate()` → `Answer`, each a public
    function a caller can run by hand (plan.md §5 rule 1, §15 #5). Zero
    context blocks return `INSUFFICIENT_CONTEXT_TEXT` with
    `insufficient_context=True`, `Usage.provider="none"` and **no model
    call**; `Answer.truncated` and `Answer.contexts` come straight from the
    `Context`. After each call the provider's reported prompt tokens are
    reconciled against the local estimate and a `WARNING` is logged when
    they exceed the margin (plan.md §18 F4). `context_budget(window,
    max_output, overhead, margin=)`, `insufficient_context(hits,
    min_results=, min_gap=)` (F10's relative signal; the gap check is
    opt-in, calibrated at Gate C — see above) and `load_vectors(docs, dim)` are
    module-level. `Rag.from_defaults(persist_dir)` — `FastEmbedEmbedder`
    wrapped in `BatchingEmbedder` + `CachingEmbedder` (`embeddings.sqlite`),
    `SqliteDocumentStore` (`nanorag.sqlite`), and `default_generator()`:
    every available option — Groq key, Gemini key, reachable Ollama — chained
    in that order (the Gate C recommendation), or a named
    `generator_preset`; nothing available raises `ConfigError` naming all
    three. `Rag` is exported from the top level lazily, so `import nanorag`
    still pulls no `numpy`; `from nanorag import Rag` pulls `numpy` but no
    `httpx`, `tiktoken`, `fastembed` or `onnxruntime` (asserted in a clean
    subprocess).
  - `tests/fakes.py`: `FakeGenerator` now satisfies the `Generator` protocol
    (`generate(Prompt) -> Generation`, records `Prompt`s, can raise
    scripted exceptions, exposes `context_window` / `max_output_tokens`).
  - `benchmarks/search_latency.py`: the Phase C acceptance harness — builds
    a 100k × 768 `NumpyVectorStore`, times unfiltered / 50 %-filtered /
    2 %-filtered searches single-threaded, prints p50 / p95 / max and a
    pass/fail line against 50 ms p95. Pins BLAS to one thread before
    `numpy` is imported. Dev laptop, two runs: unfiltered p50 ~12 ms, p95
    14–20 ms; filtered 50 % p95 19–44 ms; filtered 2 % p95 5–15 ms.
  - `examples/quickstart.py` (the README's five lines, runnable with a Groq
    key or with `NANORAG_PROFILE=local` and Ollama) and
    `examples/offline_fakes.py` (the whole pipeline with fakes — no model,
    key or network; run by CI under blocked sockets). Both ≤ 40 lines.
  - Tests: `tests/test_ratelimit.py` (buckets under `FakeClock`,
    `Retry-After` holds, jitter bounds, retry policy),
    `tests/test_timing.py`, `tests/test_generation_openai_compat.py` and
    `tests/test_generation_gemini.py` (recorded responses via
    `httpx.MockTransport`: a real Groq 429 body with `Retry-After` retried
    then raised, a daily-cap 429 as `QuotaExhausted`, `model_not_found`,
    401/403, 5xx, timeouts, malformed bodies, the key absent from every
    `repr()` / `str()`), `tests/test_generation_fallback.py` (quota
    exhaustion, a provider error and a retired model name each fall back
    Groq → Gemini over real clients and record `usage.provider="gemini"`;
    `AuthError` does not), `tests/test_pipeline.py` (ingest → answer end to
    end with fakes and no network; zero retrieved chunks → the defined
    answer with no model call; truncation and the relative signal
    propagate; `Rag.query()` reproduced from public functions alone;
    persist → reopen reproduces retrieval; re-ingest drops stale vectors;
    `default_generator` / `from_defaults` wiring), `tests/test_examples.py`,
    and new package-level import-cost invariants.
  - 585 tests in the default suite, 100% coverage on `ratelimit.py`,
    `pipeline.py`, `generation/` and `observability/` (99% on `src/`
    overall, same pre-existing gaps).
