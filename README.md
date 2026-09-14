# Nano RAG

A small, readable, production-capable retrieval-augmented generation **engine**
— not a framework. The goal is that you can read the whole thing in an
afternoon, run it against your own documents, and operate it for free.

> **Status: Phase B, session B4 (corpus to index — embeddings). Phase B is
> now feature-complete.** The package installs, lints, type-checks, tests and
> builds a wheel on Linux and Windows. Beyond Phase A's primitives, B1's
> loaders/cleaning, B2's chunkers and B3's storage (`SqliteDocumentStore`,
> `NumpyVectorStore`), it now turns text into vectors: `FastEmbedEmbedder`
> (local ONNX via `fastembed`, no torch), a persistent `EmbeddingCache` keyed
> by model + normalised text (survives process restarts — a re-embed of
> unchanged text is always free), and `BatchingEmbedder` to bound memory on
> large batches — all fully offline after the first model download, no keys.
> There is no retrieval or generation code yet. See [`plan.md`](plan.md) §9
> for the phased roadmap and [`ROADMAP.md`](ROADMAP.md) for the condensed
> version.

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

Once Phase C lands, the quickstart will be:

```bash
uv add "nanorag[local]"
```

```python
from nanorag import Rag

rag = Rag.from_defaults(persist_dir=".nanorag")   # local embeddings, best available generator
rag.ingest_path("docs/", glob="**/*.md")
answer = rag.query("How does the retriever filter?", k=8)
print(answer.text)
```

Ingest is fully offline in every configuration. Only the single generation
request per query touches the network — and pointing the generator at Ollama
removes that too.

## Limits (planned, stated up front)

- **Single-process only through 1.0.** The in-memory vector index and the
  rate limiter are per-process. Multi-worker / multi-instance / serverless
  deployments are out of scope; pointing the store at Qdrant (Phase G) does
  not by itself change that. See `plan.md` "Deployment scope".
- **English-only default embedder** (`bge-base-en-v1.5`). Non-English corpora
  need a multilingual model and lose recall on the default.
- **Corpus ceiling ≈ 200k–500k chunks** on a 32 GB / 8 GB-VRAM laptop.
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
