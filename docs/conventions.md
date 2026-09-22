# Conventions

The house style every module follows, per `plan.md` (the project's internal
design doc — not part of this repo). This reuses the sibling project
(`ML From Scratch`) toolchain
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
guess), `nanorag.pages` / `nanorag.title` (PDF/HTML source metadata),
`nanorag.heading_path` (`MarkdownChunker`), `nanorag.dedup_group` /
`nanorag.duplicate_of` (near-duplicate detection, session E2 — a document
opts a chunk-level check in by setting the group key; a flagged chunk is
kept, tagged with the winner's id, never dropped).

## Determinism & hashing

- `content_hash` and `chunk_id` normalise text the same way (NFC + LF line
  endings) via one shared helper in `hashing.py`, then sha256-truncate.
- `chunk_id` hashes the chunk's **own** normalised text, not the document
  hash, so an edit rotates only the ids of chunks that actually changed.
- Hashes are deterministic across processes, platforms and `PYTHONHASHSEED`.

## Optional loaders and cleaning (Phase G, session G1)

- `PdfLoader` and `HtmlLoader` are ordinary `Loader` implementations in
  `nanorag.loaders`; `.pdf`, `.html` and `.htm` are in `DEFAULT_LOADERS`.
  Their `pypdf` / `selectolax` imports happen inside `load()`, never at module
  import time. This keeps `.txt`/`.md` zero-extra and makes missing optional
  dependencies fail at the use boundary with `ConfigError` naming the exact
  install command.
- Parse/decode/resource failures are `LoaderError`. `DirectoryLoader` catches
  exactly that type and records one failed file without stopping the walk;
  `ConfigError` still propagates because an uninstalled requested component is
  a deployment problem, not malformed input.
- PDF pages are joined with `"\n\f\n"`. A later chunk's offsets are exact into
  the extracted `Document.text`; they do not claim page-layout coordinates,
  because `pypdf` reflows positioned glyphs. The PDF loader bounds input bytes,
  page count and extracted characters.
- HTML extraction excludes script/style/noscript/template/SVG/canvas content
  and preserves block boundaries. Site-specific navigation is not guessed at
  by the loader: cleaning remains a separate, explicit pure-text step.
- `strip_boilerplate()` removes only caller-declared whole lines and exact
  lines repeated at PDF page edges. It defaults to requiring three pages and
  leaves every retained line untouched. Offsets after cleaning, like after any
  cleaner, address the cleaned text passed to the chunker.

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
`rag.retriever`/`rag.reranker` alone, same as `query()`. `self.retriever`
itself is a plain, overridable attribute since session F2 (mirroring
`self.reranker`) — `DenseRetriever` by default, or a `Bm25Retriever`/
`HybridRetriever` wired in by hand.

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

## Hybrid retrieval (Phase F, session F2)

- `Retriever` (`retrieval/base.py`) is a `Protocol`: `retrieve(query, k,
  filter) -> list[ScoredChunk]`, written now that a second and third
  implementation exist (plan.md §15 #1) — `DenseRetriever` (Phase C),
  `Bm25Retriever` and `HybridRetriever` (session F2). `Rag.retriever` is a
  plain overridable attribute defaulting to `DenseRetriever`, the same
  pattern `Rag.reranker` already established for `IdentityReranker`.
- **`Bm25Retriever` is SQLite FTS5 with a capability check, not an
  assumption.** `SqliteDocumentStore` tries to create a `chunks_fts`
  virtual table at open time and records whether it worked as
  `fts5_available`; `AFTER INSERT`/`AFTER DELETE` triggers on `chunks` (with
  `PRAGMA recursive_triggers = ON`, needed for the trigger to fire on a
  foreign-key-cascaded delete too) keep it in sync automatically, so no
  write-path code has to know it exists. When FTS5 is unavailable,
  `Bm25Retriever` falls back to an in-memory Okapi BM25 over
  `docs.iter_chunks()`, recomputed per call rather than indexed — zero new
  dependencies either way (FTS5 ships in SQLite itself; the fallback is
  `re` + the already-core `numpy`).
- **`HybridRetriever` fuses any two or more retrievers by Reciprocal Rank
  Fusion** — rank-only, never raw score, because dense and BM25 scores sit
  on incomparable scales. `candidate_k` is retrieval's own "retrieve wide,
  keep narrow" knob, the same idea `Rag.retrieve_k` applies to reranking.
- **Enabled by default only if it beats the Phase D baseline** (plan.md §9
  Phase F Acceptance) — measured with `benchmarks/hybrid_sweep.py` against
  the same corpus and thresholds every other Phase D/F sweep uses. BM25
  alone and RRF hybrid both clear the bar by a wide margin on ranking
  metrics, but neither is wired into `from_defaults()`'s default: unlike
  F1's reranker (a model-loading cost), this is a genuine, currently
  unmitigated regression in abstention quality (see below) —
  `docs/evaluation.md` "Hybrid retrieval" has the full numbers and the
  one-liner to wire either in by hand.
- **Abstention generalises the same way F1 already forced for reranker
  scores**: judged on `self.retriever`'s own hits, never the reranked ones,
  and `min_score`/`min_gap` are only meaningful on *that retriever's* score
  scale. A non-dense `self.retriever` (BM25, RRF) is exactly as
  uncalibrated against the measured dense floor as a reranker's score is —
  confirmed by the F2 sweep, which measured `false_abstain_rate = 1.000`
  for every hybrid row before disabling `min_score` for them.

## MMR, parent-document expansion, query transforms (Phase F, session F3)

- Three composable wrappers over any `Retriever`, all opt-in, none wired
  into `from_defaults()`: `MmrRetriever` (diversify by Maximal Marginal
  Relevance — relevance from the wrapped retriever's own score,
  min-max-normalised; diversity from the chunks' own embeddings, fetched
  by id via `NumpyVectorStore.get_vectors`), `ParentExpandingRetriever`
  (widen each hit to its neighbouring chunks, re-sliced from the owning
  `Document.text` — never a concatenation of the narrow chunks' own text,
  which would double-count chunker overlap), and `MultiQueryRetriever`
  (retrieve for several phrasings of one question via a `QueryTransform`,
  fused by the same `retrieval.hybrid.rrf_fuse` `HybridRetriever` uses —
  factored out as one shared function specifically so the two RRF users
  cannot drift apart, plan.md §5 rule 1).
- **Composition order is a real constraint, not just a suggestion, for the
  two that change chunk identity or content.** `MmrRetriever` looks up
  vectors by a candidate's *current* chunk id, so it must sit *inside*
  (closer to the base retriever than) `ParentExpandingRetriever`, whose
  synthetic expanded chunks have a different id and no vector-store entry
  of their own. Both modules' docstrings state this explicitly, in each
  other's terms, so the constraint is visible from whichever one a reader
  opens first.
- **`QueryTransform` is the one deliberate exception to "the only network
  call in the whole engine is the single generation request per query"**
  (plan.md §4): `LLMQueryTransform` needs a generator call *before*
  retrieval starts. `IdentityQueryTransform` (no expansion, a true
  pass-through) is `MultiQueryRetriever`'s default for exactly that
  reason — the extra call only ever happens when a caller explicitly asks
  for it.
- **A failing `QueryTransform` degrades to the original query alone**
  (`QueryTransformError`, logged at `WARNING`) — the same "an optional
  stage never fails the query" contract `rerank_hits` and `Bm25Retriever`'s
  FTS5 fallback already uphold.
- **Enabled by default only if it beats the Phase D baseline** (plan.md §9
  Phase F Acceptance) — measured with `benchmarks/f3_sweep.py`. **MMR is
  this phase's first genuine negative result**: every `lambda_mult` tested
  lands within noise of the dense baseline on `nanorag-docs`, because the
  corpus rarely puts near-duplicate chunks in one query's top candidates
  for MMR to trade away in the first place. Parent-document expansion is a
  real but modest win (`ndcg@5` +0.025 to +0.037) that measures "does the
  shown span cover the gold quote" more than "did retrieval rank better" —
  not defaulted on for two named, documented costs (context-budget
  pressure from wider chunks; overlapping-hit redundancy `ContextBuilder`'s
  exact-text dedup does not catch), not a ranking-quality objection.
  `docs/evaluation.md` "MMR, parent-document expansion, query transforms"
  has the full numbers and the mechanism explanation.

## Public types

`Answer`, `Citation`, `Usage` and `Timings` are frozen since `v0.1.0`
(Gate C, plan.md §18 F11): adding, removing or retyping a field is a
breaking change and bumps the minor while `0.x`. The three machine-readable
status signals — `Answer.insufficient_context`, `Answer.truncated`,
`Usage.provider` (who actually served, after fallback) — are part of that
contract: a caller never parses `text` to learn any of them.

## Abstention

`insufficient_context()` is a tunable heuristic, never a guarantee. Its
thresholds live on `self.retriever`'s own score scale — the default
embedder's cosine scale for the default `DenseRetriever` — so `Rag()` ships
every optional check off and only `from_defaults()` sets the floor measured
for the default embedder/retriever pair. Anything that changes the
embedder, or swaps in a non-dense retriever (`Bm25Retriever`,
`HybridRetriever` — plan.md §9 Phase F session F2), re-measures before it
changes what `min_score`/`min_gap` mean (the Gate C measurement for the
default pair is recorded in the function docstring and README Limits; the
F2 sweep found no such floor yet exists for BM25/RRF scores —
`docs/evaluation.md` "Hybrid retrieval").

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
