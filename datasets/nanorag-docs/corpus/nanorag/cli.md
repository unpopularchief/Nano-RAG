# Command line

`nanorag ingest | query | inspect` — the library's read and write paths as a
command, nothing more. Installed as the `nanorag` console script by the
package (`uv run nanorag …` inside this repository); also runnable as
`python -m nanorag.cli`. `nanorag eval` arrives with the evaluation harness
(Phase D, session D3).

```bash
nanorag ingest docs/ --glob "**/*.md"          # offline: local embeddings, no key
nanorag query "Where do API keys come from?"   # one generation call
nanorag query "…" --json | jq .answer.citations
nanorag inspect                                # what the index holds; loads no model
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
  | `5` | the store: index built with a different embedding model, no index to inspect | `StoreError`, `IndexModelMismatch` |

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

## Schema stability

The payload keys above are the contract from the Phase D release (`v0.3.0`,
Gate D): from then on a key is only added, never renamed or removed, without
a minor version bump (the project is `0.x`). `answer` follows the stricter
freeze already on `Answer` itself.
