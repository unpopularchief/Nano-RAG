# Providers

The two halves of a RAG pipeline have opposite cost profiles, so they get
opposite answers. Embeddings are called once per chunk and open models are at
parity with closed ones — so they run **locally**. Generation is called once
per query and the quality gap is large — so it uses **free-tier APIs**, with a
local fallback.

## Embeddings — local ONNX, no torch

Run via [`fastembed`](https://github.com/qdrant/fastembed) + `onnxruntime`.
Unmetered, offline, private, deterministic on a pinned build. Install with
`uv add "nanorag[local]"`. `Rag.from_defaults()` raises a `ConfigError` naming
the exact install command if no embedder is available.

| Model | Dim | Size | Notes |
| --- | --- | --- | --- |
| `bge-base-en-v1.5` *(default candidate)* | 768 | ~440 MB | English-only. CPU does hundreds of chunks/sec. The final default is picked by measurement in Phase D, not reputation. |
| `bge-small-en-v1.5` | 384 | ~130 MB | Faster, smaller index. English-only. |
| `bge-m3` | 1024 | ~2.2 GB | Multilingual. Needed for non-English corpora. |
| `Qwen3-Embedding-0.6B` | 1024 | ~1.2 GB | Multilingual candidate. |

**The default embedder is English-only.** Non-English input degrades silently
on `bge-*-en`; use `bge-m3` and expect a larger index.

Reserve the 8 GB of laptop VRAM for **generation**, not embeddings — CPU
embedding is ample at this corpus scale.

Model files are checksummed against committed expected hashes on first
download before the cache is trusted. A model swap is a deliberate, visible
change to those hashes.

## Generation — free-tier APIs, Ollama offline

One `openai_compat.py` client, parameterised by `base_url` + `model` +
`api_key_env`, covers every OpenAI-wire-format service. `gemini.py` is a
second client for Gemini's own schema.

| Role | Choice | Free tier | Data-usage terms |
| --- | --- | --- | --- |
| **Primary** | Groq — `llama-3.3-70b-versatile`, `openai/gpt-oss-120b` | ~30 RPM, ~1K req/day, no card. Returns `Retry-After` on 429. | See Groq's current terms. |
| **Fallback** | Gemini Flash | Generous free tier. Used when Groq's daily cap is hit. | **Google's free tier may train on submitted prompts.** Paid does not. |
| **Offline** | Ollama — `qwen2.5:7b-instruct` or `llama3.1:8b` | Local, unlimited. Q4_K_M ≈ 4.7 GB weights + ~2 GB KV → fits 8 GB VRAM at 8–16k context. | Nothing leaves the machine. **The fully-private path.** |
| **Rerank** (Phase F) | Jina `jina-reranker-v2-base-multilingual`, or a local ONNX cross-encoder | 1M tokens free, 100 RPM. | **Non-commercial key.** The local cross-encoder keeps the pipeline offline. |

Local Ollama models have far smaller context windows than the API models — the
token budget is read from the generator, never hard-coded.

## Keys

Environment variables only: `GROQ_API_KEY`, `GEMINI_API_KEY`, `JINA_API_KEY`.
Never written to disk, never logged, never in `repr()` or exception text.
`.env` support is a three-line optional read in the CLI, not a dependency.
CI never holds a key — the default suite is fully offline against fakes.

## Fallback

`from_defaults()` picks the generator by what is available: Groq key → Groq;
else Gemini key → Gemini; else a reachable Ollama → Ollama; else a
`ConfigError` naming all three. It prints which it chose. Automatic fallback
between generators triggers on `QuotaExhausted`, `ProviderError` **and** a
retired model name; `Answer.usage` records which provider actually served the
query. The rate limiter is per-process — run one process per key.
