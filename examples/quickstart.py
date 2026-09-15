"""Quickstart — ingest a folder, ask a question, print the answer and sources.

The README's five lines, runnable three ways (plan.md §17). Ingest is local
and offline in every case; only the one generation call differs::

    GROQ_API_KEY=...        uv run python examples/quickstart.py   # free key
    NANORAG_PROFILE=local   uv run python examples/quickstart.py   # Ollama, offline

Needs the ``[local]`` extra for embeddings (``uv add "nanorag[local]"``).
For a run that needs no model and no key at all, see ``offline_fakes.py``.
"""

import logging
import sys

from nanorag import Rag

# A real answer can hold characters Windows' cp1252 console can't encode.
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
logging.basicConfig(level=logging.INFO, format="%(name)s: %(message)s")

rag = Rag.from_defaults(persist_dir=".nanorag")
report = rag.ingest_path("docs/", glob="**/*.md")
print(f"ingested {len(report.loaded)} documents, skipped {len(report.skipped)}")

question = sys.argv[1] if len(sys.argv) > 1 else "Where do API keys come from?"
answer = rag.query(question, k=6)

print(f"\nQ: {question}\nA: {answer.text}\n")
print(
    f"served by {answer.usage.provider} ({answer.usage.model}), "
    f"{answer.usage.total_tokens} tokens, {answer.timings.total_ms:.0f} ms"
)
print(f"insufficient={answer.insufficient_context} truncated={answer.truncated}")
print("\nsources in the prompt (label, score, file):")
for label, hit in enumerate(answer.contexts, start=1):
    doc = rag.docs.get_document(hit.chunk.doc_id)
    print(f"  [{label}] {hit.score:.3f}  {doc.source_uri if doc else hit.chunk.doc_id}")
rag.close()
