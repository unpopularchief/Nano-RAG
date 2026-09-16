# Command line

`nanorag ingest | query | inspect | eval | sync` — the library's read and
write paths, its evaluation harness, and its durability tool, as a command.
Installed as the `nanorag` console script by the package (`uv run nanorag …`
inside this repository); also runnable as `python -m nanorag.cli`.

```bash
nanorag ingest docs/ --glob "**/*.md"          # offline: local embeddings, no key
nanorag query "Where do API keys come from?"   # one generation call
nanorag query "…" --json | jq .answer.citations
nanorag inspect                                # what the index holds; loads no model
nanorag eval datasets/nanorag-docs/dev.jsonl   # offline retrieval eval; --thresholds gates
nanorag sync docs/ --glob "**/*.md"            # dry run: what would add/update/delete
nanorag sync docs/ --glob "**/*.md" --apply    # reconcile the index for real
```

## Contract

- **stdout is the result, stderr is everything else.** Log lines (`-v`),
  warnings and error messages never touch stdout, so `--json` output can be
  piped straight into a parser.
- **`--json` prints one JSON object** — the payload documented below. The
  text output is a rendering of the same payload, never a superset of it.
- **Exit codes follow the error hierarchy** (`nanorag.errors`):

  | Code | Meaning | Raised as |
  | --- | --- | --- |
  | `0` | success — including a query the engine answered with "I don't know" | |
  | `1` | any other `NanoRagError` | e.g. `RetrievalError` |
  | `2` | usage error: bad arguments, an ingest root that is not a directory, `-k 0`, a `--filter` that is not a JSON object | `argparse` |
  | `3` | configuration: no generator available, the `[local]` extra missing, an unknown profile or setting | `ConfigError` |
  | `4` | the provider: rejected key, spent quota, rate limit or transient failure that survived the retries and the fallback chain, retired model | `ProviderError` and subclasses |
  | `5` | the store: index built with a different embedding model, no index to inspect, `sync --apply` would delete past `--max-delete-fraction` | `StoreError`, `IndexModelMismatch` |
  | `6` | `eval --thresholds` only: a metric fell below a committed threshold (the report is still printed) | |

  A malformed dataset or thresholds file (`EvaluationError`) exits `1`.

  An exception that is not a `NanoRagError` is a bug and propagates with its
  traceback (the interpreter exits `1`).
- **Keys stay in the environment.** `GROQ_API_KEY`, `GEMINI_API_KEY`, … are
  read from the environment as everywhere else in the library. A `.env` file
  in the working directory (or `--env-file PATH`) is read first as a
  convenience: `KEY=VALUE` lines, `#` comments, an optional `export`, quotes
  around the value; a variable already set in the real environment always
  wins; nothing is ever printed or logged. Absent file, nothing happens.

## Common options

Every subcommand takes these (after the subcommand name):

| Option | Effect |
| --- | --- |
| `--persist-dir DIR` | where the index lives: `nanorag.sqlite` + `embeddings.sqlite` (default `settings.persist_dir`, `.nanorag`) |
| `--profile NAME` | a named settings profile — `local` forces Ollama (default `$NANORAG_PROFILE`) |
| `--embedding-model MODEL_ID` | `fastembed` model id (default `settings.embedding_model`, `BAAI/bge-base-en-v1.5`) |
| `--env-file PATH` | the `KEY=VALUE` file to export (default `.env`, ignored when absent) |
| `--json` | print the payload instead of text |
| `-v` / `-vv` | `INFO` / `DEBUG` logging on stderr (default `settings.log_level`) |

The three settings flags are overrides on top of the normal precedence
(`docs/conventions.md` → Configuration): explicit flag → `NANORAG_*` env →
profile → `[tool.nanorag]` in `pyproject.toml` → defaults.

## `nanorag ingest ROOT`

Walk `ROOT`, load every matching file, chunk, embed locally, persist. Fully
offline — no generator is constructed, so no key is read and no server is
probed. Re-running over unchanged files re-embeds nothing (the embedding
cache); a changed file replaces its chunks.

| Option | Effect |
| --- | --- |
| `--glob PATTERN` | gitignore-style pattern relative to `ROOT` (default `**/*`; `*` stays within a path segment, `**` crosses them) |
| `--ignore PATTERN` | exclude entirely; repeatable |

One file failing to load never stops the walk: it is listed under `failed`
and the exit code is still `0`. A `ROOT` that is not a directory is a usage
error (`2`).

```json
{
  "persist_dir": ".nanorag",
  "root": "docs",
  "glob": "**/*.md",
  "ignore": [],
  "chunks": 7,
  "loaded":  [{"doc_id": "a4df3bb9…", "source_uri": "conventions.md"}],
  "skipped": [{"source_uri": "image.png", "reason": "unrecognised extension"}],
  "failed":  [{"source_uri": "corrupt.txt", "reason": "…"}]
}
```

`source_uri` is the path relative to `ROOT` with POSIX separators, the same
value the index records. `skipped` holds sources never attempted
(unrecognised extension, symlink, ignored); `failed` holds ones a loader
raised on. `chunks` is the number written in this run.

## `nanorag query QUESTION`

Retrieve, budget, fence, generate — `Rag.query` exactly, then print the
answer and where it came from. Only the one generation call touches the
network; with `--generator ollama` (or `--profile local`) nothing does.

| Option | Effect |
| --- | --- |
| `-k N` | chunks to retrieve (default `8`) |
| `--filter JSON` | metadata pre-filter, a JSON object in the `store/filters.py` grammar, e.g. `'{"source_uri": {"$prefix": "docs/"}}'` |
| `--generator auto\|groq\|gemini\|ollama` | which provider; `auto` chains every available one in that order (default `settings.generator_preset`) |

An empty index is not an error — the engine abstains with no model call and
a warning on stderr pointing at `nanorag ingest`.

```json
{
  "question": "Where do API keys come from?",
  "k": 8,
  "filter": null,
  "answer": {
    "text": "Keys come from environment variables only. [3]",
    "citations": [
      {"label": 3, "chunk_id": "…", "doc_id": "…", "source_uri": "conventions.md",
       "start_char": 3580, "end_char": 5515}
    ],
    "contexts": [
      {"chunk": {"chunk_id": "…", "doc_id": "…", "ordinal": 0, "text": "…",
                 "start_char": 0, "end_char": 1983, "token_count": 480, "metadata": {}},
       "score": 0.604, "source": "dense"}
    ],
    "insufficient_context": false,
    "truncated": false,
    "usage": {"provider": "groq", "model": "openai/gpt-oss-120b",
              "prompt_tokens": 2400, "completion_tokens": 91, "total_tokens": 2491,
              "cost_usd": 0.0},
    "timings": {"embed_ms": 0.0, "retrieve_ms": 150.2, "rerank_ms": 0.0,
                "context_ms": 3.1, "generate_ms": 890.4, "total_ms": 1045.0}
  },
  "sources": [
    {"label": 1, "score": 0.604, "chunk_id": "…", "doc_id": "…",
     "source_uri": "conventions.md"}
  ]
}
```

`answer` is `Answer.to_dict()` — the public type frozen since `v0.1.0`, so
its fields change only with a minor version bump. Its three status signals
mean what they do in the library: `insufficient_context` (retrieval fell
below the floor; with no usable context at all the model was not called and
`usage.provider` is `"none"`), `truncated` (blocks were dropped to fit the
window), `usage.provider` (who actually served, after fallback).
`contexts` are exactly the chunks that were in the prompt, in label order —
`[n]` in `text` refers to `contexts[n-1]` — and `sources` resolves each
label to its `source_uri`, which `Answer` itself carries only by `doc_id`.
`citations` holds the markers that resolved; a marker that did not is
dropped and logged, never fabricated.

## `nanorag inspect`

Describe the index without loading a model or constructing a generator:
opens the SQLite store directly. A missing index is `StoreError` (exit `5`);
nothing is created.

```json
{
  "persist_dir": ".nanorag",
  "index": {"model_id": "BAAI/bge-base-en-v1.5", "dim": 768},
  "documents": 2,
  "chunks": 7,
  "sources": [
    {"doc_id": "a4df3bb9…", "source_uri": "conventions.md", "content_hash": "…",
     "chunks": 3, "metadata": {}}
  ]
}
```

`index` is `null` until the first embeddings are written. `sources` is
ordered by `source_uri`.

## `nanorag eval DATASET.jsonl`

Grade the pipeline on a frozen dataset (`docs/evaluation.md` has the
metrics, `datasets/nanorag-docs/README.md` the format). By default a
**retrieval** run — the corpus is embedded locally into an in-memory index
(the embedding cache under `--persist-dir` makes a repeat, or a chunk-size
sweep, cheap), every question is retrieved and the ranking graded against
the gold spans. Offline and keyless. With `--answers` it becomes an
**answer** run: one generation call per item through the configured
generator.

| Option | Effect |
| --- | --- |
| `--corpus DIR` | the corpus directory (default `corpus/` next to `DATASET`) |
| `--ks K,K,…` | cut-offs to report (default `1,3,5,10`) |
| `--chunk-tokens N`, `--overlap-tokens N` | the `RecursiveChunker` to grade (default: the library defaults) |
| `--min-score X` | the abstention floor (default: the measured floor for the default embedder, none for any other) |
| `--thresholds FILE` | check the report against a `thresholds.json`; a regression exits `6` |
| `--answers` | answer run instead of retrieval; `-k N` chunks per query, `--generator` picks the provider |

```json
{
  "report": {
    "dataset": "nanorag-docs",
    "kind": "retrieval",
    "config": {"embedder": "BAAI/bge-base-en-v1.5", "dim": 768,
               "chunker": "RecursiveChunker", "target_tokens": 128, "overlap_tokens": 64,
               "chunks": 839, "min_score": 0.55, "min_gap": 0.0, "ks": [1, 3, 5, 10]},
    "n_items": 341,
    "n_answerable": 317,
    "metrics": {"mrr": 0.659, "recall@1": 0.544, "precision@1": 0.546, "hit_rate@1": 0.546,
                "ndcg@1": 0.546, "recall@5": 0.836, "…": "…",
                "false_abstain_rate": 0.003, "abstain_rate": 0.417},
    "items": [
      {"id": "readme-011", "answerable": true, "tags": ["readme", "generation"],
       "top_score": 0.71, "flagged_insufficient": false, "retrieved": ["…"],
       "first_relevant_rank": 1, "metrics": {"mrr": 1.0, "recall@1": 1.0, "…": "…"}}
    ]
  },
  "thresholds": {"path": "datasets/nanorag-docs/thresholds.json", "passed": true,
                 "violations": []}
}
```

`thresholds` is `null` without `--thresholds`; a violation is
`{"metric", "value", "floor", "baseline"}`. An answer run's `config` adds
`generator`, `model` and `k`; its `metrics` are `answer_rate`,
`abstain_rate`, `cited_rate`, `citation_validity`, `citation_precision`,
`token_f1`, `mean_total_tokens`, `mean_generate_ms` and `completed_rate`
(an answer run records a provider failure on the item and carries on, and
stops after a `QuotaExhausted`, an `AuthError` or ten failures in a row —
so a rate-limited free tier still yields a report over what completed);
each item row carries the answer `text`, its `citations`, provider and
tokens, or an `error`.

## `nanorag sync ROOT`

Reconcile the index against a fresh walk of `ROOT` (Phase E, `Rag.sync_path`):
a `source_uri` on disk but not in the index is an addition, one in both with
a changed `content_hash` is an update, one in the index but no longer on
disk is a deletion. **Dry run by default** — nothing is written, the payload
is the plan; pass `--apply` to execute it. Offline like `ingest`: no
generator is constructed. `added`/`updated` go through the same change
detection as `ingest` (a re-sync of a large, mostly-unchanged corpus is
cheap), and `deleted` cascades through SQLite and tombstones the in-memory
index in the same call as `Rag.delete_document`.

| Option | Effect |
| --- | --- |
| `--glob PATTERN`, `--ignore PATTERN` | same as `ingest` |
| `--apply` | execute the plan (default: report only) |
| `--max-delete-fraction FRACTION` | `--apply` refuses if more than this fraction of the index's documents would be deleted (default `0.5`) — a mistyped `ROOT` or an unmounted volume must not silently empty the corpus |

```json
{
  "added": ["c.md"],
  "updated": ["a.md"],
  "unchanged": ["b.md"],
  "deleted": ["old.md"],
  "skipped": [],
  "failed": [],
  "applied": false,
  "over_delete_guard": false,
  "persist_dir": ".nanorag",
  "root": "docs",
  "glob": "**/*.md",
  "ignore": []
}
```

`over_delete_guard` is `true` whenever `deleted` exceeds `--max-delete-fraction`
of the index's document count from before this sync — reported on a dry run
too (a warning on stderr), so the condition is visible before `--apply` is
ever passed. With `--apply`, tripping the guard raises instead of writing
anything: a sync either fully succeeds (`added`/`updated`/`deleted` all
applied) or fully does not.

## Schema stability

The `ingest`/`query`/`inspect`/`eval` payload keys above are the contract
from the Phase D release (`v0.3.0`, Gate D): from then on a key is only
added, never renamed or removed, without a minor version bump (the project
is `0.x`). `answer` follows the stricter freeze already on `Answer` itself.
`sync`'s payload (Phase E) is not yet frozen — it stays informally unstable
until Gate E, the same posture `IngestReport`/`SyncReport` have in
`docs/conventions.md`.
