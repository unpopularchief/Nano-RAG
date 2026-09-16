# Roadmap

Condensed from [`plan.md`](plan.md) §9. Eight phases, twenty-three sessions.
Each session ends at a checkpoint; **each phase ends at a review gate where
work stops** before the next phase begins. Version tags are targets, not a
contiguous release list.

| Phase | Version | Goal | Status |
| --- | --- | --- | --- |
| **A — Foundation** | `v0.0.1` | Repo installs, lints, type-checks, tests, builds a wheel on Linux + Windows. Core data types. | **done; Gate A passed** |
| **B — Corpus to index** | `v0.0.4` | Real files → durable, searchable vectors. Fully offline. | **done; Gate B passed** |
| **C — First answers (MVP)** | `v0.1.0` | Documents in, cited answer out — free key or fully offline. First public release. | **done — `v0.1.0`, Gate C passed** |
| **D — Trust** | `v0.3.0` | Citations resolving to text spans, a CLI, quality as numbers in CI (≥ 200-item eval set). | **done — `v0.3.0`, Gate D passed** |
| **E — Durability** | `v0.4.0` | Re-ingesting a changed corpus is correct and cheap. | **done — `v0.4.0`, Gate E passed** |
| **F — Quality** | `v0.6.0` | Beat the Phase D baseline with evidence — reranking, BM25/hybrid, MMR. Negative results published. | F1 done — F2, F3 next |
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
- **🚦 Gate C** ✅ — the MVP, `v0.1.0`. Decisions (plan.md §9 checklist),
  each locked against the working read path: `Answer` / `Citation` /
  `Usage` / `Timings` frozen as shipped in C3; `from_defaults()` keeps
  auto-chaining every available generator; the abstention signal measured
  with the real embedder on two corpora — the relative gap does not
  separate answerable from unrelated questions, an absolute top-1 floor
  does, so `min_gap` stays off, a `min_score` floor was added, and
  `from_defaults()` sets `0.55` for `bge-base-en-v1.5` only; F1–F4
  confirmed in place (F4 via the 15 % margin + `usage` reconciliation, no
  `tokenizers` dependency); preset defaults checked against the live
  catalogues (OpenRouter's had been retired and was replaced); the
  three-way quickstart run — fakes in CI, plus the Groq and Ollama legs
  recorded in the Gate C notes.

## Phase D sessions

- **D1** ✅ — `citations/parser.py`. `parse_citations(text, context,
  get_document=)` scans generated text for `[n]`-shaped markers (one label
  per bracket — `[1][2]`, never `[1,2]` or `[1-3]`), resolves each against
  the `Context.blocks` the answer was actually prompted with, and returns a
  `CitationReport`: deduplicated, label-ordered `Citation`s plus
  `total_markers`/`resolved_markers`/`validity_rate`. An unresolvable
  marker (out of range, zero, or simply never in this context) is dropped
  from the citations but still counted towards the validity rate — never
  silently ignored. Metadata propagates doc → chunk → context → citation
  through the existing chain (`ScoredChunk.chunk.doc_id`/`start_char`/
  `end_char`, `get_document().source_uri`); a citation whose chunk's
  document is missing from the store raises `StoreError` rather than
  fabricating one (should be impossible — SQLite commits before a chunk is
  ever placed in a prompt). Wired into `Rag.query`: `Answer.citations` is
  now `parse_citations(...).citations` instead of always `()`, and a
  marker that didn't resolve is logged as a warning (the "report a
  validity rate" half, matching `_reconcile_usage`'s pattern rather than
  adding a new `Answer` field — `Answer` is frozen as of `v0.1.0`).
  Checkpoint met: a 4-document, 7-question fixture corpus with one
  deliberately hallucinated marker resolves 21/22 markers (≥ 95%); a 300
  example Hypothesis oracle checks resolution against a plain-Python
  reference over arbitrary marker streams and block counts.
- **D2** ✅ — `cli/`: `nanorag ingest | query | inspect`, installed as the
  `nanorag` console script (also `python -m nanorag.cli`). Each subcommand
  is one module with the same shape — `run()` returns a JSON-serialisable
  payload, `render()` turns it into text — so `--json` and the text output
  are the same data. stdout is the result and nothing else; logs, warnings
  and errors go to stderr. Exit codes follow the error hierarchy: `0`
  success (an "I don't know" included), `1` other `NanoRagError`, `2`
  usage, `3` `ConfigError`, `4` `ProviderError`, `5` `StoreError`; a
  non-`nanorag` exception propagates as the bug it is. `ingest` is fully
  offline — a new `generation.NullGenerator` fills the generator slot so no
  key is read and no server probed; `inspect` opens the SQLite store
  directly and never creates one. `query --json` is `Answer.to_dict()` (the
  frozen public type) plus `sources` resolving each context label to its
  file; `--filter` takes the `store/filters.py` grammar as JSON;
  `--generator` maps onto the existing preset lookup in `Settings`. The
  "three-line optional `.env` read" from plan.md §4: `KEY=VALUE` lines
  exported only where unset, never printed. Checkpoint met: the payload
  schemas and exit codes are documented in `docs/cli.md` and asserted key
  for key in `tests/test_cli.py`. The live run (Groq, Ollama, a rejected
  key) exposed a D1 gap — `openai/gpt-oss-120b` writes its markers as
  fullwidth `【3】`, which the parser neither resolved nor counted — fixed
  by accepting `【n】` alongside `[n]`. `nanorag eval` lands with the
  harness in D3.
- **D3** ✅ — `evaluation/` and the dataset. `datasets/nanorag-docs/`: 19
  frozen documents (7 of the project's own docs at D2 plus 12 permissively
  licensed package READMEs, attributed) and `dev.jsonl` with **341
  hand-written items** — 317 answerable, gold given as a verbatim quote
  that is resolved to character offsets at load time (missing or
  ambiguous quotes fail loudly), so one dataset grades every chunker; 24
  unanswerable, half off-topic and half on-topic-but-uncovered.
  `retrieval_metrics.py` (Recall/Precision/hit-rate@k, MRR, nDCG@k with
  novelty gains so overlapping chunks cannot push it past 1),
  `answer_metrics.py` (citation validity and precision against gold,
  abstention, SQuAD token-F1), `runner.py` (`evaluate_retrieval` offline
  through the real `Rag`; `evaluate_answers` one call per item, records
  provider failures per item and stops on quota/auth or ten in a row),
  `report.py` (`EvalReport`, `thresholds.json` = baseline + tolerance).
  `nanorag eval` (exit 6 on a threshold regression), `pytest -m eval` +
  `.github/workflows/eval.yml` (nightly, one fixed image, offline),
  `benchmarks/eval_sweep.py`. **The sweep picked the chunker default**:
  128/64 over the plan's 512/64 (Recall@5 0.836 vs 0.694; the 7B Ollama
  leg's answer rate 0.918 vs 0.785, citation precision 0.69 vs 0.61). The
  free-tier API legs of the answer eval did not complete (Groq's daily
  token cap, Gemini's rate limit) — a Gate D item, on a subset.
  Baseline committed in `thresholds.json` with a 0.05 band (≈ 2 SE at
  n = 317); everything in `docs/evaluation.md`.
- **🚦 Gate D** ✅ — `v0.3.0`. Reviewed against plan.md §9's Phase D block:
  all three checkpoints hold (≥ 95% marker resolution, stable CLI schema
  and exit codes, a 341-item eval set with committed thresholds), and the
  eval gate was re-verified on GitHub's own CI runner (not just locally)
  at the post-D3 CRLF-fix commit (`0192ce3`) — green, reproducing
  `thresholds.json` within tolerance
  (Recall@5 0.836, Recall@10 0.880, MRR 0.658, nDCG@5 0.698). **The
  free-tier API-model answer leg (Groq/Gemini) is scoped out of the gate,
  not a blocker** — both hit quota/rate limits during D3 and neither had
  recovered by this session; the chunk-default decision already rests on
  the D3 sweep's retrieval metrics plus the completed Ollama leg pointing
  the same direction, and re-running the API leg needs a `--limit`/
  `--tags` flag on `nanorag eval` that doesn't exist yet — tracked as a
  follow-up (see CHANGELOG `[0.3.0]`), not reopened as Phase D work.
  Classifier bumped `3 - Alpha` → `4 - Beta`.

## Phase E sessions

- **E1** ✅ — Change detection, `delete_document` cascade, tombstone mask,
  `compact()`. `Rag.ingest()` now compares an incoming document's
  `content_hash` against what is already stored for its `doc_id` and, on a
  match, skips chunking, embedding and every store write entirely — a
  document-level shortcut that sits above the existing chunk-level reuse
  (`CachingEmbedder` keyed by each chunk's own text) rather than replacing
  it, so a genuinely changed document still re-embeds only the chunks
  whose text moved. `Rag.delete_document(doc_id)` cascades through SQLite
  (chunks, embeddings — `ON DELETE CASCADE`) and tombstones the same chunk
  ids in `NumpyVectorStore` in one call, so a deleted chunk is unreachable
  from both `retrieve()` and a store rebuild from that instant;
  `Rag.compact()` physically reclaims the tombstoned rows. The tombstone
  mask and `NumpyVectorStore.delete`/`compact`, and
  `SqliteDocumentStore.delete_document`'s cascade, were already built in
  Phase B3 by design (plan.md: "designed into the Phase B schema even
  though the API lands in Phase E") — E1 is the facade wiring plan.md
  scoped here, plus the tests proving the whole path end to end. Checkpoint
  met: deleted chunks vanish from search and SQLite in the same call to
  `delete_document`, confirmed by a pipeline-level test independent of the
  store-level tests already covering the primitives.
- **E2** ✅ — `sync_path()` reconciliation and near-duplicate detection.
  `Rag.sync_path(root, apply=False)` walks a directory and classifies every
  `source_uri` against the store's current documents into added / updated
  / unchanged / deleted, by the same `content_hash` comparison `ingest()`
  already does; `apply=True` executes the plan (`ingest()` for added and
  updated, `delete_document()` for deleted) and refuses to run at all —
  writing nothing, not even the safe adds — if the deletions exceed
  `max_delete_fraction` (default 50%) of the store's document count, so a
  mistyped root or an unmounted volume cannot silently empty the corpus
  (plan.md §18 F9). `over_delete_guard` on the returned `SyncReport` is
  computed on a dry run too, so the condition is visible before `apply`
  is ever passed. New `nanorag sync ROOT [--apply]
  [--max-delete-fraction F]` CLI command, offline like `ingest` — landed
  in the same session as `sync_path()` itself, the same norm D3 set for
  `nanorag eval`. Near-duplicate detection (plan.md §18 F13): a document
  opts a chunk into the check via `metadata["nanorag.dedup_group"]`; a
  chunk found ≥ 0.97 cosine-similar to an already-indexed chunk in the
  same group — a different document, `exclude` keeps a document from ever
  matching its own current or prior chunks — is kept, never dropped
  (dropping would rotate every later chunk's id for no reason and break a
  citation already pointing at it), but tagged
  `metadata["nanorag.duplicate_of"]` with the winner's chunk id. Bounded
  cost, the F13 requirement: a document that never sets the group key is
  never compared against anything, and one that does is compared only
  within its own group (found via the existing `filter_chunk_ids`
  metadata filter, not a new index) — never against the whole corpus.
  Documented limit: comparisons run against chunks already persisted
  before this ingest call, so two never-before-seen near-duplicate
  documents landing in the same batch do not catch each other.
- **🚦 Gate E** ✅ — `v0.4.0`. Reviewed against plan.md §9's Phase E block:
  the E1/E2 checkpoints hold, and both Acceptance lines were verified
  literally rather than taken as implied by the mechanism. "No orphan
  rows in any table after a full add/edit/delete cycle" — a new test
  drives one `sync_path(apply=True)` through an add, an edit and a delete
  together and queries `chunks`/`embeddings` directly for rows pointing
  at a document or chunk that no longer exists; two adjacent,
  previously-untested Phase E Tests-list items were closed alongside it
  (editing one document leaves every other document's chunk ids and
  ordinals untouched; a crash between the SQLite commit and the
  in-memory index update strands the in-memory index but never SQLite,
  and a fresh rebuild from SQLite alone is exact). "Phase D eval numbers
  unchanged by a re-sync" — the real, already-embedded committed corpus
  was resynced against itself unedited and re-measured inside the
  existing `-m eval` threshold test (reusing the same embedded index
  rather than a second cold pass, which would have doubled `eval.yml`'s
  CI runtime): every metric came back bit-identical. See CHANGELOG
  `[0.4.0]` for the full list.

## Phase F sessions

- **F1** ✅ — `rerank/`: `base.py` (the `Reranker` protocol — `rerank(query,
  hits, top_n) -> list[ScoredChunk]`), `identity.py` (`IdentityReranker`,
  the default — a pure slice, no re-scoring), `jina.py` (`JinaReranker`,
  the Jina Reranker API, unit-tested via `MockTransport` — no
  `JINA_API_KEY` was available this session, so its error mapping is
  inferred from the general API shape, not confirmed live, and it was not
  part of the measured comparison below), `local_cross_encoder.py`
  (`LocalCrossEncoderReranker`, offline via `fastembed`'s ONNX
  cross-encoders, `Xenova/ms-marco-MiniLM-L-6-v2` by default). `Rag` gained
  `reranker=`/`retrieve_k=`: `k` is what a query keeps, `retrieve_k`
  (defaults to `k`) is how many candidates the retriever fetches before
  reranking — "retrieve wide, keep narrow." A reranker that raises
  `RerankError` degrades to the retriever's own order (logged at
  `WARNING`) rather than failing the query.
  - **Measured against the real committed corpus/thresholds**
    (`benchmarks/rerank_sweep.py`, `docs/evaluation.md` "Reranking"): the
    local cross-encoder clears the Phase F Acceptance bar by a wide
    margin — `ndcg@5` 0.698 → 0.827 at `retrieve_k=32` (≈ 2.6× the 0.05
    tolerance band), `recall@1` 0.544 → 0.721 — with gains plateauing
    between `retrieve_k=32` and `64` (+0.003 `ndcg@5` for 1.7× the
    reranking time), which is why 32 was picked.
  - **A real bug found and fixed by the sweep, not a hypothetical**:
    `insufficient_context()`'s `min_score`/`min_gap` floors are
    calibrated against the *embedder's* score scale (F10); reusing a
    reranker's differently-scaled score for the same check silently
    inflated `false_abstain_rate` from 0.003 to 0.10–0.13 in the first
    (uncorrected) run. Fixed in `Rag.query()` and `evaluate_retrieval()`
    to judge abstention on the retriever's dense hits always, reranking a
    separate copy for the actual context/ranking metrics; new
    module-level `pipeline.rerank_hits()` shares the "rerank, degrade on
    failure" logic between `Rag.retrieve()`, `Rag.query()` and the eval
    runner (plan.md §5 rule 1). Confirmed by the corrected sweep: every
    reranked row's `false_abstain_rate` now matches the baseline exactly.
  - **Not wired into `Rag.from_defaults()`'s default**, despite clearing
    the bar: `LocalCrossEncoderReranker` loads its ONNX model eagerly at
    construction, and `from_defaults()` also builds the `Rag` behind
    `nanorag ingest`, which plan.md §4 requires to stay network-free.
    Flipping the default safely needs lazy reranker construction (load on
    first `retrieve()`/`query()`, not at `Rag()`/`from_defaults()` time) —
    a named follow-up, scoped out of F1 rather than silently deferred.
    `Rag()`'s own bare default stays `IdentityReranker` (zero
    dependencies) either way.

## Out of scope through 1.0

Multi-worker / multi-instance / serverless deployment · a shared rate limiter ·
learned rerankers · Graph RAG · agentic retrieval · query planners · fine-tuning
· an HTTP server · a GUI.
