# Roadmap

Condensed from [`plan.md`](plan.md) §9. Eight phases, twenty-three sessions.
Each session ends at a checkpoint; **each phase ends at a review gate where
work stops** before the next phase begins. Version tags are targets, not a
contiguous release list.

| Phase | Version | Goal | Status |
| --- | --- | --- | --- |
| **A — Foundation** | `v0.0.1` | Repo installs, lints, type-checks, tests, builds a wheel on Linux + Windows. Core data types. | **A1–A3 done; Gate A next** |
| **B — Corpus to index** | `v0.0.4` | Real files → durable, searchable vectors. Fully offline. | not started |
| **C — First answers (MVP)** | `v0.1.0` | Documents in, cited answer out — free key or fully offline. First public release. | not started |
| **D — Trust** | `v0.3.0` | Citations resolving to text spans, a CLI, quality as numbers in CI (≥ 200-item eval set). | not started |
| **E — Durability** | `v0.4.0` | Re-ingesting a changed corpus is correct and cheap. | not started |
| **F — Quality** | `v0.6.0` | Beat the Phase D baseline with evidence — reranking, BM25/hybrid, MMR. Negative results published. | not started |
| **G — Reach** | `v0.7.0` | PDF/HTML loaders, external stores (Qdrant, pgvector), hosted embeddings — without touching the core. | not started |
| **H — Production** | `v1.0.0` | Structured logging, cost accounting, deployment guide, threat model, API freeze. | not started |

## Phase A sessions

- **A1** ✅ — `git init`, licence, ignore/attributes, `pyproject.toml`,
  pre-commit, CI, doc skeletons. No library code. *(push + CI-green pending a
  git remote.)*
- **A2** ✅ — `errors.py`, `hashing.py`, `types.py` + tests. Hash determinism
  verified in subprocesses under two `PYTHONHASHSEED` values; types immutable
  and validated.
- **A3** ✅ — `tokens.py`, `config.py`, `tests/fakes.py` + tests. Config
  precedence (explicit > env > profile > `pyproject.toml` > default); fakes
  ready for every later phase.
- **🚦 Gate A** — *next.* Toolchain and primitives reviewed before any I/O is
  written.

## Out of scope through 1.0

Multi-worker / multi-instance / serverless deployment · a shared rate limiter ·
learned rerankers · Graph RAG · agentic retrieval · query planners · fine-tuning
· an HTTP server · a GUI.
