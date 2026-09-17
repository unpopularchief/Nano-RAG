"""``QueryTransform``/``MultiQueryRetriever``: the F3 "optional query
transforms behind flags" checkpoint — ``IdentityQueryTransform`` is a true
no-op (byte-identical to not wrapping at all), ``LLMQueryTransform`` parses
and validates a generator's response, a failing transform degrades to the
original query alone rather than failing the call, and RRF fusion across
phrasings matches the same hand-computable arithmetic
``test_hybrid_retriever.py`` already established for fusion across
retrievers — plan.md §9 Phase F session F3."""

import logging

import pytest

from nanorag.errors import ConfigError, QueryTransformError, RetrievalError
from nanorag.hashing import chunk_id, content_hash, stable_doc_id
from nanorag.retrieval.query_transform import (
    SOURCE,
    IdentityQueryTransform,
    LLMQueryTransform,
    MultiQueryRetriever,
)
from nanorag.types import Chunk, Document, ScoredChunk
from tests.fakes import FakeGenerator, FakeRetriever


def _doc(uri: str) -> Document:
    text = f"text of {uri}"
    return Document(
        doc_id=stable_doc_id(uri),
        source_uri=uri,
        text=text,
        content_hash=content_hash(text),
    )


def _chunk(doc: Document, ordinal: int) -> Chunk:
    text = f"chunk {ordinal}"
    return Chunk(
        chunk_id=chunk_id(doc.doc_id, ordinal, text),
        doc_id=doc.doc_id,
        ordinal=ordinal,
        text=text,
        start_char=0,
        end_char=len(text),
        token_count=1,
    )


_DOC = _doc("qt.txt")
_CHUNKS = [_chunk(_DOC, i) for i in range(4)]


def _hit(i: int, score: float, source: str = "dense") -> ScoredChunk:
    return ScoredChunk(chunk=_CHUNKS[i], score=score, source=source)


class _FakeTransform:
    """A scriptable ``QueryTransform`` for driving ``MultiQueryRetriever``."""

    def __init__(self, phrasings=None, *, error=None):
        self.phrasings = phrasings if phrasings is not None else []
        self.error = error
        self.calls: list[str] = []

    def transform(self, query: str) -> list[str]:
        self.calls.append(query)
        if self.error is not None:
            raise self.error
        return self.phrasings


# --- IdentityQueryTransform -------------------------------------------------


def test_identity_query_transform_returns_no_extra_phrasings():
    assert IdentityQueryTransform().transform("anything") == []


# --- LLMQueryTransform -------------------------------------------------------


def test_llm_query_transform_rejects_non_positive_n():
    with pytest.raises(ConfigError):
        LLMQueryTransform(FakeGenerator(), n=0)


def test_llm_query_transform_parses_one_phrasing_per_line():
    generator = FakeGenerator(["how does X work?\nwhat is X?\nexplain X"])
    phrasings = LLMQueryTransform(generator, n=3).transform("what is X")
    assert phrasings == ["how does X work?", "what is X?", "explain X"]


def test_llm_query_transform_strips_blank_lines_and_truncates_to_n():
    generator = FakeGenerator(["one\n\n  two  \nthree\nfour"])
    phrasings = LLMQueryTransform(generator, n=2).transform("q")
    assert phrasings == ["one", "two"]


def test_llm_query_transform_records_the_real_prompt():
    generator = FakeGenerator(["a phrasing"])
    LLMQueryTransform(generator, n=1).transform("what is X")
    assert "what is X" in generator.last_prompt


def test_llm_query_transform_wraps_a_provider_error():
    from nanorag.errors import TransientError

    generator = FakeGenerator([TransientError("down")])
    with pytest.raises(QueryTransformError):
        LLMQueryTransform(generator).transform("q")


def test_llm_query_transform_raises_on_a_blank_response():
    generator = FakeGenerator(["   \n\n  "])
    with pytest.raises(QueryTransformError):
        LLMQueryTransform(generator).transform("q")


# --- MultiQueryRetriever: construction ---------------------------------------


@pytest.mark.parametrize("kwargs", [{"k": 0}, {"candidate_k": 0}, {"rrf_k": 0}])
def test_non_positive_parameters_are_rejected_at_construction(kwargs):
    with pytest.raises(RetrievalError):
        MultiQueryRetriever(FakeRetriever([]), **kwargs)


def test_non_positive_k_is_rejected_at_call():
    retriever = MultiQueryRetriever(FakeRetriever([]))
    with pytest.raises(RetrievalError):
        retriever.retrieve("q", k=0)


def test_default_k_property():
    assert MultiQueryRetriever(FakeRetriever([]), k=7).k == 7


# --- the F3 checkpoint: identity is a true pass-through -----------------------


def test_default_transform_is_identity_and_result_is_byte_identical_to_wrapped():
    fake = FakeRetriever([_hit(0, 0.9), _hit(1, 0.5)])
    retriever = MultiQueryRetriever(fake)

    result = retriever.retrieve("the question", k=2, filter={"kind": "x"})

    assert result == fake.retrieve("the question", k=2, filter={"kind": "x"})
    assert all(h.source == "dense" for h in result)  # untouched, not fused
    assert len(fake.calls) == 2  # one for `retriever.retrieve`, one for the check above


def test_identity_transform_calls_the_wrapped_retriever_exactly_once():
    fake = FakeRetriever([_hit(0, 0.9)])
    MultiQueryRetriever(fake).retrieve("q", k=1)
    assert len(fake.calls) == 1
    assert fake.calls[0] == ("q", 1, None)


# --- RRF fusion across phrasings, hand-computed ------------------------------


def test_fuses_across_original_plus_extra_phrasings_by_rrf():
    # The original query and one variant each surface a different top hit;
    # fusion must combine both, same arithmetic as HybridRetriever's own
    # RRF-across-retrievers test.
    class _PhrasingAwareRetriever:
        def __init__(self):
            self.calls = []

        def retrieve(self, query, k=None, filter=None):
            self.calls.append((query, k, filter))
            if query == "original":
                return [_hit(0, 0.9), _hit(1, 0.5)]
            return [_hit(2, 0.9), _hit(0, 0.5)]

    wrapped = _PhrasingAwareRetriever()
    transform = _FakeTransform(["variant"])
    retriever = MultiQueryRetriever(wrapped, transform, rrf_k=60)

    hits = retriever.retrieve("original", k=3)

    rrf_k = 60
    expected = {
        _CHUNKS[0].chunk_id: 1 / (rrf_k + 1) + 1 / (rrf_k + 2),
        _CHUNKS[2].chunk_id: 1 / (rrf_k + 1),
        _CHUNKS[1].chunk_id: 1 / (rrf_k + 2),
    }
    got = {h.chunk.chunk_id: h.score for h in hits}
    for cid, score in expected.items():
        assert got[cid] == pytest.approx(score)
    assert [h.chunk.chunk_id for h in hits] == sorted(
        expected, key=lambda cid: -expected[cid]
    )
    assert all(h.source == SOURCE for h in hits)
    assert {c[0] for c in wrapped.calls} == {"original", "variant"}


def test_a_phrasing_identical_to_the_original_or_duplicated_is_not_retrieved_twice():
    fake = FakeRetriever([_hit(0, 0.9)])
    transform = _FakeTransform(["dup", "dup", "original query"])
    MultiQueryRetriever(fake, transform).retrieve("original query", k=1)
    queried = [call[0] for call in fake.calls]
    assert queried == ["original query", "dup"]  # "dup" once, echo of original dropped


def test_candidate_k_and_filter_are_forwarded_to_every_phrasing():
    fake = FakeRetriever([])
    transform = _FakeTransform(["variant"])
    MultiQueryRetriever(fake, transform, k=5, candidate_k=20).retrieve(
        "q", filter={"kind": "x"}
    )
    assert fake.calls[0] == ("q", 20, {"kind": "x"})
    assert fake.calls[1] == ("variant", 20, {"kind": "x"})


# --- a failing transform degrades, never fails the query ---------------------


def test_failing_transform_degrades_to_the_original_query_alone(caplog):
    fake = FakeRetriever([_hit(0, 0.9)])
    transform = _FakeTransform(error=QueryTransformError("boom"))
    retriever = MultiQueryRetriever(fake, transform)

    with caplog.at_level(logging.WARNING):
        hits = retriever.retrieve("q", k=1)

    assert hits == fake.hits[:1]
    assert len(fake.calls) == 1  # never retried per-phrasing, no phrasings survived
    assert any("query transform" in r.message.lower() for r in caplog.records)
