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

## Clients

`nanorag.generation.OpenAICompatGenerator(preset, model)` covers Groq,
OpenRouter and Ollama (and any other OpenAI-wire-format server — a new
`Preset` line, not a new file); `GeminiGenerator(model)` covers Gemini. Both
expose `context_window` and `max_output_tokens` — the pipeline budgets
context from those, never from a constant — and both turn HTTP failures into
the typed errors in `nanorag.errors`:

| Response | Error | Retried? |
| --- | --- | --- |
| 401 / 403 | `AuthError` | no — surfaced, never hidden by fallback |
| 429 naming a **daily** cap (Groq RPD/TPD, a Gemini `…PerDay…` quota) | `QuotaExhausted` | no — straight to the fallback |
| any other 429 | `RateLimitError` (carries `Retry-After`) | yes, honouring `Retry-After` |
| 404 / `model_not_found` (a retired model name) | `ProviderError` tagged `code="model_not_found"` | no — straight to the fallback |
| 5xx, timeout, connection failure | `TransientError` | yes, bounded exponential backoff with jitter |
| any other 4xx | `ProviderError` | no |

`ratelimit.RateLimiter(rpm=, tpm=, rpd=)` can be handed to a client to keep
under a tier's caps proactively; the retry loop pushes every `Retry-After`
into it as a hold. Groq's free tier at the time of writing: ~30 RPM,
~1K requests/day on the 70B model — set `rpm=30, rpd=1000` to stay polite.

## Fallback

`from_defaults()` picks the generator by what is available: Groq key → Groq;
Gemini key → Gemini; a reachable Ollama (`GET localhost:11434/v1/models`)
→ Ollama — and **chains every option it finds, in that order**, as a
`FallbackGenerator`; nothing available raises a `ConfigError` naming all
three. It logs which it chose at `INFO`. A named
`generator_preset` (`NANORAG_GENERATOR_PRESET=ollama`, or the `local`
profile) uses only that one. Fallback triggers on `QuotaExhausted`, any
`ProviderError` (a retired model name included) and a `RateLimitError` /
`TransientError` that survived the client's own retries — **not** on
`AuthError`, because a rejected key is a configuration problem you want to
see, not one to answer around from a different account. Every hand-off is
logged at `WARNING` and `Answer.usage.provider` records which member
actually served the query. The chain budgets context to the *smallest*
window among its members (Ollama's 4096 when it is in the chain), so the
prompt fits whoever ends up serving. The rate limiter is per-process — run
one process per key.

## Context windows and token counting

The token budget for retrieved context is `(window − reserved output −
prompt overhead) × (1 − margin)`, read from the generator. Counting uses
`tiktoken` cl100k — an approximation for the Llama / Gemini / Qwen
tokenizers actually in play — so the default margin is 15 %, and after
every call the provider's reported prompt tokens are compared with the
local estimate: exceeding the margin logs a `WARNING` naming the fix (pass
the generator's real tokenizer as `counter=`, or raise `budget_margin`).
**Ollama serves every model with a fixed 4096-token context by default**
(`OLLAMA_CONTEXT_LENGTH`), whatever the model card says — the `OLLAMA`
preset assumes exactly that; override `context_window=` if you raised it.
