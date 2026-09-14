"""The whole pipeline with fakes — no model download, no key, no network.

Hand-wires ``Rag`` from the test fakes so anyone can watch the read path
work end to end in a second: ingest -> exact retrieval -> budgeted, fenced
context -> a scripted "model" -> ``Answer``. Run from the repository root::

    uv run python examples/offline_fakes.py

This is also what CI runs (``tests/test_examples.py``).
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from tests.fakes import FakeEmbedder, FakeGenerator  # noqa: E402

from nanorag import Rag  # noqa: E402
from nanorag.store import SqliteDocumentStore  # noqa: E402
from nanorag.tokens import HeuristicCounter  # noqa: E402

generator = FakeGenerator(["Keys come from environment variables only. [1]"])
rag = Rag(
    embedder=FakeEmbedder(dim=256),
    generator=generator,
    docs=SqliteDocumentStore(":memory:"),
    counter=HeuristicCounter(warn=False),  # no tokenizer download either
)
report = rag.ingest_path("docs/", glob="**/*.md")
print(f"ingested {len(report.loaded)} documents")

question = "Where do API keys come from?"
answer = rag.query(question, k=3)
print(f"\nQ: {question}\nA: {answer.text}  (served by {answer.usage.provider})\n")
for label, hit in enumerate(answer.contexts, start=1):
    doc = rag.docs.get_document(hit.chunk.doc_id)
    print(f"  [{label}] {hit.score:.3f}  {doc.source_uri if doc else '?'}")
print("\n--- exact prompt the generator saw, nonce fence and all (first 400) ---")
print(generator.last_prompt[:400])
