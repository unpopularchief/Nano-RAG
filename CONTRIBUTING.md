# Contributing

Nano RAG is built **one phase at a time**, per `plan.md` §9 (the project's
internal design doc — not part of this repo). Each phase ends at a review
gate where work stops. Nothing from a later phase is scaffolded ahead of
time. See [`ROADMAP.md`](ROADMAP.md) for the condensed, public version of
the phase plan.

## The dependency line

- **A new core dependency needs a written justification in the PR.** Core is
  three packages (`numpy`, `httpx`, `tiktoken`) and stays that way.
- **A new provider needs a `MockTransport` test and zero core changes.** One
  OpenAI-compatible client covers Groq, OpenRouter, Together, Ollama, LM Studio
  and vLLM. Each provider file is ≤ 200 lines.
- **AGPL is never a default.** Licence review is part of the dependency
  justification.

## The two anti-framework rules

1. **The facade owns no logic.** `Rag.query()` is a readable ~30-line
   composition of public functions. Anything it does, a user can do by calling
   those functions directly. A test reproduces `Rag.query()` from public
   functions alone.
2. **Protocols, not base classes.** Every extension point is a
   `typing.Protocol`. A `Protocol` is written when the *second* implementation
   exists — not before. No registry, no metaclass, no dynamic dispatch you
   cannot follow by reading.

## Conventions

Full detail in [`docs/conventions.md`](docs/conventions.md). In short: `src/`
layout, one thing per file, NumPy-style docstrings, ruff `E,F,I,UP,B,NPY,D` at
88 columns, mypy strict on `src/`, frozen dataclasses for core types, keys from
environment variables only (never logged, never in `repr()`).

## Workflow

1. One phase session per PR (see `plan.md` §9 and the PR template).
2. Implement, following the conventions.
3. Tests: deterministic, offline, no API key. Slow/networked tests go behind
   the `local` / `provider` / `eval` / `integration` markers.
4. Update `CHANGELOG.md` (Keep a Changelog format) in the *same* PR. Update
   `README.md` / `ROADMAP.md` if status or capabilities changed.
5. `pre-commit` mirrors CI exactly — run `pre-commit run --all-files`.

Conventional commits. CI must be green on Ubuntu and Windows across Python
3.11 / 3.12 / 3.13 before merge.
