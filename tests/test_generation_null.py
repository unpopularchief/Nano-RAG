"""``generation/null.py``: the null object a write-only pipeline wires in
place of a provider — satisfies the protocol, budgets sanely, refuses to
generate."""

import pytest

from nanorag import Rag
from nanorag.errors import ConfigError
from nanorag.generation import Generator, NullGenerator
from nanorag.prompting import PromptBuilder
from nanorag.store import SqliteDocumentStore
from nanorag.tokens import HeuristicCounter
from tests.fakes import FakeEmbedder
from tests.test_pipeline import _doc


def test_null_generator_satisfies_the_protocol_and_refuses_to_generate():
    generator = NullGenerator()
    assert isinstance(generator, Generator)
    assert generator.provider == "none" and generator.model == "none"
    assert generator.context_window > generator.max_output_tokens > 0
    assert repr(generator) == "NullGenerator()"
    with pytest.raises(ConfigError, match="no generator was configured"):
        generator.generate(PromptBuilder().build("q", _empty_context()))


def _empty_context():
    from nanorag.pipeline import _EMPTY_CONTEXT

    return _EMPTY_CONTEXT


def test_rag_with_a_null_generator_ingests_but_a_query_fails_loudly():
    rag = Rag(
        embedder=FakeEmbedder(dim=64),
        generator=NullGenerator(),
        docs=SqliteDocumentStore(":memory:"),
        counter=HeuristicCounter(warn=False),
    )
    assert rag.ingest([_doc("a.txt", "alpha beta gamma")]) == 1
    assert rag.retrieve("alpha")[0].chunk.doc_id == _doc("a.txt", "x").doc_id
    # Retrieval found context, so the pipeline reaches the generator — and
    # gets a ConfigError rather than a silent "I don't know".
    with pytest.raises(ConfigError):
        rag.query("alpha?")
    rag.close()
