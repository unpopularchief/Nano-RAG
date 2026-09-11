# Roadmap

Condensed from [`plan.md`](plan.md) §9. Eight phases, twenty-three sessions.
Each session ends at a checkpoint; **each phase ends at a review gate where
work stops** before the next phase begins. Version tags are targets, not a
contiguous release list.

| Phase | Version | Goal | Status |
| --- | --- | --- | --- |
| **A — Foundation** | `v0.0.1` | Repo installs, lints, type-checks, tests, builds a wheel on Linux + Windows. Core data types. | **done; Gate A passed** |
| **B — Corpus to index** | `v0.0.4` | Real files → durable, searchable vectors. Fully offline. | **B1–B2 done; B3 next** |
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
- **B3** — `store/sqlite_docs.py` + `store/numpy_store.py` — *next.*
- **B4** — `embeddings/` (local ONNX via fastembed, cache, batching).
- **🚦 Gate B** — storage schema and offline ingest path reviewed before
  anything reads from them.

## Out of scope through 1.0

Multi-worker / multi-instance / serverless deployment · a shared rate limiter ·
learned rerankers · Graph RAG · agentic retrieval · query planners · fine-tuning
· an HTTP server · a GUI.
