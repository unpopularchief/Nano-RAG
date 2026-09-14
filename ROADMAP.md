# Roadmap

Condensed from [`plan.md`](plan.md) §9. Eight phases, twenty-three sessions.
Each session ends at a checkpoint; **each phase ends at a review gate where
work stops** before the next phase begins. Version tags are targets, not a
contiguous release list.

| Phase | Version | Goal | Status |
| --- | --- | --- | --- |
| **A — Foundation** | `v0.0.1` | Repo installs, lints, type-checks, tests, builds a wheel on Linux + Windows. Core data types. | **done; Gate A passed** |
| **B — Corpus to index** | `v0.0.4` | Real files → durable, searchable vectors. Fully offline. | **done; Gate B passed** |
| **C — First answers (MVP)** | `v0.1.0` | Documents in, cited answer out — free key or fully offline. First public release. | not started |
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

## Out of scope through 1.0

Multi-worker / multi-instance / serverless deployment · a shared rate limiter ·
learned rerankers · Graph RAG · agentic retrieval · query planners · fine-tuning
· an HTTP server · a GUI.
