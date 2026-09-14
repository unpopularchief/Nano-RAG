# Nano RAG

A small, readable, production-capable retrieval-augmented generation **engine**
— not a framework. The goal is that you can read the whole thing in an
afternoon, run it against your own documents, and operate it for free.

> **Status: Phase C feature-complete (C1–C3 done); Gate C review pending —
> `v0.1.0` ships when it closes.** The package installs, lints, type-checks,
> tests and builds a wheel on Linux and Windows. It loads real files
> (`loaders/`), cleans and chunks them (`cleaning/`, `chunking/`), embeds
> them locally (`embeddings/` — `fastembed`/ONNX, no torch, cached and
> batched), persists text and vectors (`store/`), retrieves with exact
> cosine top-*k* and a metadata **pre**-filter grammar (`retrieval/`,
> `store/filters.py`), fits the hits into a token budget as `[n]`-labelled,
> nonce-fenced blocks (`context/`, `prompting/`), and now **answers**:
> `generation/` has one client for every OpenAI-wire-format service (Groq,
> OpenRouter, Ollama, …) and one for Gemini, with typed errors, retries and
> a fallback chain; `ratelimit.py` guards the single metered call; and
> `Rag` (`pipeline.py`) is the ~30-line facade that composes it all. The
> quickstart below works. Citations resolve to text spans in Phase D. See
> [`plan.md`](plan.md) §9 for the phased roadmap and [`ROADMAP.md`](ROADMAP.md)
> for the condensed version.

## What it will be

- **Traceable.** Every answer traces back to exact chunks, exact scores, exact
  prompt bytes.
- **Correct before clever.** Exact vector search, deterministic ids,
  transactional ingest, measured recall — before ANN, agents or query planners.
- **Measured.** Retrieval quality is a number in CI. A regression in Recall@5
  fails the build.
- **Free to run.** Local ONNX embeddings (no torch, no GPU needed) plus
  free-tier generation APIs (Groq, Gemini) with Ollama as a first-class offline
  path. The full pipeline runs on a laptop with no spend and no keys.
- **Thin at the edges.** Providers are ~150-line HTTP clients behind a
  `Protocol`. Core logic never imports a provider.

## What it will not be

Not an agent framework. Not a universal integration layer. No chain/graph/DSL
runtime — composition is Python function calls. Not a distributed system.

## Runs free

```bash
uv add "nanorag[local]"        # local ONNX embeddings; ~440 MB model on first run
export GROQ_API_KEY=...        # free at console.groq.com — or GEMINI_API_KEY, or run Ollama
```

```python
from nanorag import Rag

rag = Rag.from_defaults(persist_dir=".nanorag")   # local embeddings, best available generator
rag.ingest_path("docs/", glob="**/*.md")
answer = rag.query("Where do API keys come from?", k=8)
print(answer.text)
for hit in answer.contexts:                        # exactly the chunks that were in the prompt
    print(hit.score, rag.docs.get_document(hit.chunk.doc_id).source_uri)
```

`from_defaults()` picks the generator by what is available — a Groq key,
then a Gemini key, then a reachable Ollama — and **chains every option it
finds** as a fallback, so a spent daily cap or a retired model name on one
provider is answered by the next; `answer.usage.provider` says which one
served. Ingest is fully offline in every configuration. Only the single
generation request per query touches the network — and pointing the
generator at Ollama (`NANORAG_PROFILE=local`) removes that too.

The same flow with no model, no key and no network at all:
`uv run python examples/offline_fakes.py`. The README flow as a script:
`examples/quickstart.py`.

When retrieval returns nothing usable, the answer is a fixed "I don't have
enough information in the provided documents to answer that." with
`answer.insufficient_context = True` and **no model call** — the engine says
"I don't know" rather than inventing something. When retrieved chunks were
dropped to fit the model's window, `answer.truncated = True`. Neither is
ever silent.

## Limits (stated up front)

- **Single-process only through 1.0.** The in-memory vector index and the
  rate limiter are per-process. Multi-worker / multi-instance / serverless
  deployments are out of scope; pointing the store at Qdrant (Phase G) does
  not by itself change that. See `plan.md` "Deployment scope".
- **English-only default embedder** (`bge-base-en-v1.5`). Non-English corpora
  need a multilingual model and lose recall on the default.
- **Corpus ceiling ≈ 200k–500k chunks** on a 32 GB / 8 GB-VRAM laptop.
  Exact search at 100k × 768 measures ~12 ms p50 and 14–20 ms p95 across
  runs, single-threaded, on the dev laptop (`benchmarks/search_latency.py`;
  the plan's acceptance line is < 50 ms p95). **Filtered queries are
  slower** — a metadata filter is an unindexed `json_extract` scan plus an
  id mask; a 50 % filter measured 19–44 ms p95 at the same size.
- **Context budgeting uses an approximate tokenizer** (`tiktoken` cl100k)
  for Llama/Gemini/Qwen models, with a 15 % safety margin and a warning when
  the provider's reported prompt tokens exceed it. Pass the generator's real
  tokenizer as `counter=` for exact budgets.
- **Ollama serves every model with a 4096-token context** by default
  (`OLLAMA_CONTEXT_LENGTH`); the budget is read from the generator, so
  answers over Ollama see far less context than over Groq.
- Free generation tiers may train on submitted prompts (Google's free tier
  does). The Ollama path is the fully-private option. See `docs/providers.md`.
- A citation proves **provenance, not truth**.

## Development

```bash
uv sync --extra dev
uv run ruff check . && uv run ruff format --check .
uv run mypy src
uv run pytest -q
uv build
```

CI runs the same on Ubuntu and Windows across Python 3.11 / 3.12 / 3.13. No CI
job holds an API key.

The default `pytest -q` run is offline, keyless and model-free. Real-embedder
tests (`nanorag.embeddings.local.FastEmbedEmbedder`) are opt-in and download
the pinned ONNX model on first run:

```bash
uv sync --extra dev --extra local
uv run pytest -q -m local
```

## Licence

MIT. See [`LICENSE`](LICENSE). AGPL dependencies are never a default
(`pymupdf` would only ever appear as an opt-in `[pdf-agpl]` extra).
