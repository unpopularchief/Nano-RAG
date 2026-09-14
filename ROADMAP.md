# Roadmap

Condensed from [`plan.md`](plan.md) §9. Eight phases, twenty-three sessions.
Each session ends at a checkpoint; **each phase ends at a review gate where
work stops** before the next phase begins. Version tags are targets, not a
contiguous release list.

| Phase | Version | Goal | Status |
| --- | --- | --- | --- |
| **A — Foundation** | `v0.0.1` | Repo installs, lints, type-checks, tests, builds a wheel on Linux + Windows. Core data types. | **done; Gate A passed** |
| **B — Corpus to index** | `v0.0.4` | Real files → durable, searchable vectors. Fully offline. | **done; Gate B passed** |
| **C — First answers (MVP)** | `v0.1.0` | Documents in, cited answer out — free key or fully offline. First public release. | **feature-complete (C1–C3); Gate C review pending** |
| **D — Trust** | `v0.3.0` | Citations resolving to text spans, a CLI, quality as numbers in CI (≥ 200-item eval set). | not started |
| **E — Durability** | `v0.4.0` | Re-ingesting a changed corpus is correct and cheap. | not started |
| **F — Quality** | `v0.6.0` | Beat the Phase D baseline with evidence — reranking, BM25/hybrid, MMR. Negative results published. | not started |
| **G — Reach** | `v0.7.0` | PDF/HTML loaders, external stores (Qdrant, pgvector), hosted embeddings — without touching the core. | not started |
| **H — Production** | `v1.0.0` | Structured logging, cost accounting, deployment guide, threat model, API freeze. | not started |

## Phase A sessions

- **A1** ✅ — `git init`, licence, ignore/attributes, `pyproject.toml`,
  pre-commit, CI, doc skeletons. No library code.
- **A2** ✅ — `errors.py`, `hashing.py`, `types.py` + tests. Hash determinism
  verified in subprocesses under two `PYTHONHASHSEED` values; types immutable
  and validated.
- **A3** ✅ — `tokens.py`, `config.py`, `tests/fakes.py` + tests. Config
  precedence (explicit > env > profile > `pyproject.toml` > default); fakes
  ready for every later phase.
- **🚦 Gate A** ✅ — passed. CI green on all 6 matrix cells.

## Phase B sessions

- **B1** ✅ — `loaders/` (text, markdown, directory w/ glob+ignore),
  `cleaning/normalize.py`, `IngestReport`/`LoadIssue`, fixture corpus in
  `tests/data/`. `load_path()` produces identical `content_hash` regardless
  of platform (relative POSIX `source_uri`, LF-normalising hash). Symlinks
  always skipped, never followed. Gitignore-style glob matching
  (`loaders/globbing.py`) — `*` stays within a path segment, `**` crosses
  segments.
- **B2** ✅ — `chunking/` (`base`, `fixed`, `recursive` default, `markdown`).
  Every chunk's offsets reconstruct the source text and the union of chunk
  spans covers the document with no gaps, regardless of overlap — a
  Hypothesis property test verifies this across 500 generated inputs on all
  three chunkers. Markdown chunks carry a `nanorag.heading_path` and never
  straddle a heading boundary.
- **B3** ✅ — `store/sqlite_docs.py` + `store/numpy_store.py`. Documents,
  chunks and embeddings persist in SQLite (one transaction per document,
  cascading delete, `IndexModelMismatch` if a second embedding model/dim is
  used); `NumpyVectorStore` is an in-memory exact-cosine index (upsert,
  tombstone-delete, compact, top-k search) rebuilt from the SQLite
  embeddings on reopen, reproducing search results exactly.
- **B4** ✅ — `embeddings/` (`base` protocol, `local` ONNX via fastembed,
  `cache`, `batching`). `FastEmbedEmbedder` raises `ConfigError` naming the
  install command if `[local]` is missing, defaults to single-threaded ONNX
  execution for determinism, and L2-normalises at the boundary.
  `CachingEmbedder` keys on `(model_id, sha256(normalised text))`, not
  `chunk_id`, so a re-ingest re-embeds only chunks whose text actually
  changed. Checkpoint met: 10k chunks embedded locally, a second run against
  the same on-disk cache makes zero model calls.
- **🚦 Gate B** ✅ — passed. Reviewed against plan.md §9's Phase B block item
  by item; the phase-level Acceptance bullet ("10k chunks ingest and persist
  with no network access at all, verified by blocking sockets in the test;
  reopen + search reproduces pre-restart results exactly; a second identical
  ingest performs zero embedding work") had never been tested end-to-end —
  each session proved its own component in isolation. Closed by adding
  `tests/test_gate_b_offline_ingest.py`: a 200-document, 10,680-chunk
  synthetic corpus through the real `DirectoryLoader` → `FixedChunker` →
  `FastEmbedEmbedder`(cached, batched) → `SqliteDocumentStore` +
  `NumpyVectorStore` pipeline, inside a `blocked_sockets()` context
  (`tests/conftest.py`, patches `socket.socket.connect`/`connect_ex`) —
  reopen reproduces search exactly, and a second identical run makes zero
  calls to the real model. Also added the 5 MB fixture test the Phase B
  Tests bullet calls for (generated at test time, not committed).

## Phase C sessions

- **C1** ✅ — `retrieval/dense.py` + `store/filters.py`. `DenseRetriever`
  embeds the query, resolves the filter to chunk ids in SQLite, searches the
  NumPy index over those ids only, and hydrates hits back into `Chunk`s. The
  filter grammar (`$eq`/`$ne`/`$gt`/`$gte`/`$lt`/`$lte`/`$in`/`$nin`/
  `$prefix`/`$exists`, `$and`/`$or`/`$not`; `chunk_id`/`doc_id`/`ordinal`/
  `source_uri` columns plus any metadata key, chunk-then-document) compiles
  to bound-parameter SQL and is Hypothesis-checked against a plain-Python
  reference evaluator. `NumpyVectorStore.search` now breaks ties by chunk
  id (row order differs between a live index and one rebuilt on reopen) and
  no longer copies the whole matrix per query (100k × 768 unfiltered:
  ~5 ms p95 locally, down from ~175 ms). Checkpoint met: retrieval matches a
  brute-force NumPy reference exactly, filtered and unfiltered; a filter
  matching nothing returns `[]`, not an error; a test proves post-filtering
  silently loses results.
- **C2** ✅ — `context/builder.py`, `prompting/templates.py` + `fencing.py`.
  `ContextBuilder(budget_tokens, counter=)` takes hits in rank order, drops
  repeated chunk ids / repeated normalised texts / empty chunks, and takes
  blocks in order until the first one that does not fit — so the context is
  always a rank-order prefix (no holes), a chunk is never split, and
  `Context.truncated` records any drop. The budget is checked on the
  rendered, joined text under the real counter, not a per-block sum, so it
  holds for tokenizers that merge across block boundaries. Returns the
  blocks and the `chunk_id -> label` map. `PromptBuilder` assembles the
  system prompt (an instruction hierarchy: fenced text is data, answer only
  from it, cite `[n]`, abstain with a fixed `INSUFFICIENT_CONTEXT_TEXT`) and
  a user message of the context inside a per-request nonce fence followed
  by the question. `make_nonce(avoid=)` never returns a value present in
  the fenced text or the question, and `fence()` refuses a body containing
  its nonce, so a closing fence is unforgeable from inside. Checkpoint met:
  a 500-example Hypothesis test with adversarial chunk sizes (empty,
  whitespace-only, one character, far over budget, duplicates) and four
  counters including super- and sub-additive ones asserts the context
  never exceeds the budget.
- **C3** ✅ — `ratelimit.py`, `generation/`, `pipeline.py`,
  `observability/timing.py`, `benchmarks/`, first `examples/`.
  `RateLimiter` (token buckets for RPM/TPM/RPD plus a `Retry-After` hold),
  `Backoff` (bounded exponential, full jitter, `Retry-After` honoured
  verbatim) and `call_with_retry` (retries 429 / 5xx / timeouts only; never
  any other 4xx, never `QuotaExhausted`). `generation/`: the `Generator`
  protocol (`generate(Prompt) -> Generation`, plus `context_window` and
  `max_output_tokens` so the budget is read from the generator, never
  hard-coded); `OpenAICompatGenerator` — one `httpx` client for every
  OpenAI-wire-format service via a `Preset` (`GROQ`, `OPENROUTER`, `OLLAMA`
  shipped); `GeminiGenerator` for Gemini's own schema; both map HTTP
  failures to the typed errors (401/403 → `AuthError`; 429 →
  `QuotaExhausted` when the body names a daily cap, else `RateLimitError`
  with `Retry-After`; 404 → `ProviderError` tagged `model_not_found`; 5xx /
  timeouts → `TransientError`) and never expose the key in `repr()` or an
  exception. `FallbackGenerator` chains providers: quota exhaustion, a
  provider error, a retired model name, or a rate limit / transient failure
  that survived the client's retries all hand off to the next member (a
  rejected key does not — that is a configuration problem to surface), and
  `Usage.provider` records who served. `pipeline.py`: `Rag` — ~30 lines of
  composition; `context_budget()` (window − output − overhead, × (1 −
  margin)), `insufficient_context()` (the F10 relative signal: count floor
  plus an opt-in top-1-vs-rest gap) and `load_vectors()` are public
  functions it merely sequences. Zero context blocks return the fixed
  `INSUFFICIENT_CONTEXT_TEXT` with `insufficient_context=True` and no model
  call. `from_defaults()` wires local ONNX embeddings (cached, batched) +
  SQLite under `persist_dir` and chains every available generator (Groq →
  Gemini → Ollama). `observability/timing.py`: `Timer` → `Answer.timings`.
  `benchmarks/search_latency.py`: the acceptance harness — 100k × 768
  unfiltered p95 14–20 ms across runs, single-threaded, on the dev laptop
  (limit 50 ms); filtered 50 % 19–44 ms, 2 % 5–15 ms. `examples/quickstart.py` (the README
  flow) and `examples/offline_fakes.py` (no model, key or network; run in
  CI). Checkpoint: the README quickstart works with fakes in CI
  (`tests/test_examples.py`); the Groq and Ollama legs need a key / a local
  server and are part of the Gate C review, not the default suite.
- **🚦 Gate C** — the MVP. **Pending.** Decision checklist in `plan.md` §9:
  freeze `Answer`; confirm `from_defaults()` auto-chaining (implemented as
  recommended); lock the `insufficient_context` thresholds against a real
  corpus (`min_gap` ships disabled — only the count floor is active until
  measured); confirm F1–F4 (F4 taken as the margin + `usage` reconciliation
  route, no `tokenizers` dependency); the design-hold review on a real
  corpus; and the three-way quickstart run (Groq key / Ollama / fakes).

## Out of scope through 1.0

Multi-worker / multi-instance / serverless deployment · a shared rate limiter ·
learned rerankers · Graph RAG · agentic retrieval · query planners · fine-tuning
· an HTTP server · a GUI.
