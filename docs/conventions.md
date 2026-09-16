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
Reserved `nanorag.*` metadata key namespace: `nanorag.mime` (a loader's own
guess), `nanorag.heading_path` (`MarkdownChunker`), `nanorag.dedup_group` /
`nanorag.duplicate_of` (near-duplicate detection, session E2 — a document
opts a chunk-level check in by setting the group key; a flagged chunk is
kept, tagged with the winner's id, never dropped).

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
only inside the facade, that is a bug. `delete_document(doc_id)` and
`compact()` (Phase E, session E1) are the same pattern on the write side:
each is a thin sequence over `SqliteDocumentStore.delete_document` and
`NumpyVectorStore.delete`/`compact`, which already do the real work
(cascade, tombstone mask) — the facade only wires the two stores together
so a deletion is never visible in one but not the other. `sync_path()`
(session E2) composes one level up: it classifies a directory walk against
the store's current documents, then calls `ingest()` and `delete_document()`
themselves for the actual writes — it owns no write path of its own, only
the add/update/delete decision plus the delete-fraction guard. `retrieve()`
(Phase F, session F1) is the same shape again: it fetches from
`self.retriever`, hands the result to `self.reranker`, and catches exactly
one error type around that second call — a test reproduces it from
`rag.retriever`/`rag.reranker` alone, same as `query()`.

## Reranking (Phase F, session F1)

- `Reranker` (`rerank/base.py`) is a `Protocol`: `rerank(query, hits, top_n)
  -> list[ScoredChunk]`. `IdentityReranker` (a pure slice, no re-scoring) and
  `JinaReranker` are the two implementations the protocol is written at
  (plan.md §15 #1); `LocalCrossEncoderReranker` is a third.
  `IdentityReranker` is `Rag`'s default, so a caller who never touches
  reranking sees behaviour byte-identical to before Phase F.
- **retrieve-k / rerank-to-n**: `Rag.k` is what a query keeps; `Rag.retrieve_k`
  (defaults to `Rag.k`) is how many candidates the retriever fetches before
  reranking. A per-call `k` that exceeds `retrieve_k` still gets a
  correctly-sized candidate pool — `retrieve()` fetches
  `max(retrieve_k, k)`, never fewer than what was asked for.
  `retrieve_k`/`k` are the same value with the identity reranker (no
  oversampling needed for a no-op), so they cost nothing until a real
  reranker is wired in.
- **A reranker never fails the query.** `Reranker` implementations raise
  `RerankError` (never a raw provider or model exception) for any failure;
  `Rag.retrieve()` catches exactly that type and falls back to the
  retriever's own order, logging a `WARNING`. An HTTP-backed reranker
  (`JinaReranker`) translates its own `ProviderError` into `RerankError` at
  the `rerank()` boundary — the pipeline's catch stays narrow and generic.
- **Enabled by default only if it beats the Phase D baseline** (plan.md §9
  Phase F Acceptance) — measured with `benchmarks/rerank_sweep.py` against
  the same committed corpus and thresholds the chunk-size sweep used.
  Every delta, including a negative one, is recorded in
  `docs/evaluation.md`, not just the winning configuration. F1's
  `LocalCrossEncoderReranker` clears this bar by a wide margin but is
  *not* wired into `from_defaults()`'s default: it loads its model
  eagerly at construction, and `from_defaults()` also builds the `Rag`
  behind `nanorag ingest`, which must stay network-free (plan.md §4).
  Wiring it in safely needs lazy reranker construction — a named
  follow-up, not part of F1 — so today it is opt-in
  (`docs/evaluation.md` "Reranking" has the one-liner).
- **Abstention is judged on the retriever's dense hits, never the
  reranked ones** — `min_score`/`min_gap` are calibrated against the
  embedder's score scale (F10), which a reranker's score generally is
  not. `Rag.query()` and `evaluate_retrieval()` both fetch the dense hits
  separately for this check and rerank a copy for the actual context/
  ranking metrics; a real bug found by the F1 sweep before this fix
  landed, not a hypothetical.

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
`run(args, settings) -> payload` + `render(payload) -> str` +
`exit_code(payload) -> int`, so `--json` and the text output are the same
data and the JSON schema in `docs/cli.md` is the whole contract. stdout is the result and nothing else; logging,
warnings and errors go to stderr. Exit codes map onto the error hierarchy
(`ConfigError` 3, `ProviderError` 4, `StoreError` 5, other `NanoRagError`
1, usage 2, a failed eval threshold 6); an exception that is not a
`NanoRagError` propagates. No
logic lives in the CLI that a library caller cannot reach.

## Evaluation

Quality is a number, measured on a **frozen** dataset (`datasets/<name>/`:
a corpus that is never edited after its baseline is taken, `dev.jsonl`
items whose gold is a verbatim quote resolved to character offsets, so
one dataset grades every chunker alike). `evaluation/` is pure functions
over ranked lists plus a runner that drives the real `Rag`; the retrieval
run is offline and gates CI (`-m eval`, one fixed runner image,
`thresholds.json` = baseline + tolerance ≥ the set's noise floor); the
answer run needs a generator and is a manual, two-model comparison.
Nothing ships enabled without a measured delta on this set; negative
results are recorded in `docs/evaluation.md`. Never lower a floor to make
a red run green — re-baseline only after a deliberate, documented change.

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
