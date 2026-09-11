import dataclasses
import json

import pytest

from nanorag.types import (
    Answer,
    Chunk,
    Citation,
    Document,
    IngestReport,
    LoadIssue,
    ScoredChunk,
    Timings,
    Usage,
)


def make_chunk(**overrides) -> Chunk:
    kwargs = dict(
        chunk_id="c0",
        doc_id="d0",
        ordinal=0,
        text="a chunk",
        start_char=0,
        end_char=7,
        token_count=2,
    )
    kwargs.update(overrides)
    return Chunk(**kwargs)


def make_answer(**overrides) -> Answer:
    kwargs = dict(
        text="hi [1]",
        citations=(
            Citation(
                label=1,
                chunk_id="c0",
                doc_id="d0",
                source_uri="docs/a.md",
                start_char=0,
                end_char=7,
            ),
        ),
        contexts=(ScoredChunk(chunk=make_chunk(), score=0.9, source="dense"),),
        insufficient_context=False,
        truncated=False,
        usage=Usage(
            provider="fake",
            model="fake-1",
            prompt_tokens=10,
            completion_tokens=5,
            total_tokens=15,
        ),
        timings=Timings(total_ms=1.0),
    )
    kwargs.update(overrides)
    return Answer(**kwargs)


# --- immutability -----------------------------------------------------------


def test_document_is_frozen():
    doc = Document("d0", "docs/a.md", "text", "hash")
    with pytest.raises(dataclasses.FrozenInstanceError):
        doc.text = "other"


def test_metadata_is_read_only_after_construction():
    doc = Document("d0", "docs/a.md", "text", "hash", {"lang": "en"})
    with pytest.raises(TypeError):
        doc.metadata["lang"] = "fr"  # type: ignore[index]


def test_mutating_the_original_metadata_dict_does_not_leak_in():
    original = {"lang": "en"}
    doc = Document("d0", "docs/a.md", "text", "hash", original)
    original["lang"] = "fr"
    assert doc.metadata["lang"] == "en"


def test_scored_chunk_is_frozen():
    sc = ScoredChunk(chunk=make_chunk(), score=0.5, source="dense")
    with pytest.raises(dataclasses.FrozenInstanceError):
        sc.score = 0.1


# --- equality --------------------------------------------------------------


def test_equal_field_values_give_equal_instances():
    a = Document("d0", "docs/a.md", "text", "hash", {"k": 1})
    b = Document("d0", "docs/a.md", "text", "hash", {"k": 1})
    assert a == b


def test_metadata_difference_breaks_equality():
    a = Document("d0", "docs/a.md", "text", "hash", {"k": 1})
    b = Document("d0", "docs/a.md", "text", "hash", {"k": 2})
    assert a != b


# --- metadata validation --------------------------------------------------


def test_metadata_rejects_non_string_keys():
    with pytest.raises(TypeError):
        Document("d0", "docs/a.md", "t", "h", {1: "x"})  # type: ignore[dict-item]


@pytest.mark.parametrize("bad", [["a", "b"], {"nested": 1}, (1, 2), b"bytes", object()])
def test_metadata_rejects_non_scalar_values(bad):
    with pytest.raises(TypeError):
        Document("d0", "docs/a.md", "t", "h", {"k": bad})


@pytest.mark.parametrize("value", ["s", 1, 1.5, True, None])
def test_metadata_accepts_json_scalars(value):
    doc = Document("d0", "docs/a.md", "t", "h", {"k": value})
    assert doc.metadata["k"] == value


def test_metadata_rejects_non_finite_floats():
    with pytest.raises(ValueError):
        Chunk("c0", "d0", 0, "t", 0, 1, 1, {"score": float("nan")})


# --- field validation ---------------------------------------------------


@pytest.mark.parametrize("name", ["doc_id", "source_uri", "content_hash"])
def test_document_rejects_empty_required_strings(name):
    kwargs = dict(doc_id="d0", source_uri="s", text="t", content_hash="h")
    kwargs[name] = ""
    with pytest.raises(ValueError):
        Document(**kwargs)


@pytest.mark.parametrize("name", ["chunk_id", "doc_id"])
def test_chunk_rejects_empty_ids(name):
    with pytest.raises(ValueError):
        make_chunk(**{name: ""})


def test_chunk_rejects_negative_ordinal():
    with pytest.raises(ValueError):
        make_chunk(ordinal=-1)


def test_chunk_rejects_end_before_start():
    with pytest.raises(ValueError):
        make_chunk(start_char=10, end_char=3)


def test_chunk_rejects_negative_start():
    with pytest.raises(ValueError):
        make_chunk(start_char=-1, end_char=0)


def test_chunk_rejects_negative_token_count():
    with pytest.raises(ValueError):
        make_chunk(token_count=-1)


def test_chunk_allows_zero_length_span():
    chunk = make_chunk(start_char=5, end_char=5, text="")
    assert chunk.start_char == chunk.end_char


@pytest.mark.parametrize("score", [float("nan"), float("inf"), float("-inf")])
def test_scored_chunk_rejects_non_finite_score(score):
    with pytest.raises(ValueError):
        ScoredChunk(chunk=make_chunk(), score=score, source="dense")


def test_scored_chunk_rejects_empty_source():
    with pytest.raises(ValueError):
        ScoredChunk(chunk=make_chunk(), score=0.5, source="")


def test_citation_rejects_label_below_one():
    with pytest.raises(ValueError):
        Citation(
            label=0, chunk_id="c", doc_id="d", source_uri="s", start_char=0, end_char=1
        )


@pytest.mark.parametrize("name", ["chunk_id", "doc_id", "source_uri"])
def test_citation_rejects_empty_strings(name):
    kwargs = dict(
        label=1, chunk_id="c", doc_id="d", source_uri="s", start_char=0, end_char=1
    )
    kwargs[name] = ""
    with pytest.raises(ValueError):
        Citation(**kwargs)


def test_citation_rejects_bad_span():
    with pytest.raises(ValueError):
        Citation(
            label=1, chunk_id="c", doc_id="d", source_uri="s", start_char=5, end_char=1
        )


def test_usage_rejects_negative_tokens():
    with pytest.raises(ValueError):
        Usage("groq", "m", prompt_tokens=-1, completion_tokens=0, total_tokens=0)


def test_usage_rejects_empty_provider():
    with pytest.raises(ValueError):
        Usage("", "m", prompt_tokens=0, completion_tokens=0, total_tokens=0)


def test_usage_rejects_negative_cost():
    with pytest.raises(ValueError):
        Usage(
            "groq",
            "m",
            prompt_tokens=0,
            completion_tokens=0,
            total_tokens=0,
            cost_usd=-0.01,
        )


def test_usage_cost_defaults_to_zero():
    assert Usage("ollama", "llama3", 0, 0, 0).cost_usd == 0.0


def test_timings_defaults_are_zero():
    t = Timings()
    assert t.total_ms == 0.0 and t.embed_ms == 0.0


def test_timings_rejects_negative():
    with pytest.raises(ValueError):
        Timings(generate_ms=-1.0)


def test_answer_rejects_non_tuple_citations():
    with pytest.raises(TypeError):
        make_answer(citations=[])


def test_answer_rejects_non_tuple_contexts():
    with pytest.raises(TypeError):
        make_answer(contexts=[])


# --- to_dict / JSON ------------------------------------------------------


def test_answer_to_dict_is_json_serialisable():
    answer = make_answer()
    blob = json.dumps(answer.to_dict())
    restored = json.loads(blob)
    assert restored["text"] == "hi [1]"
    assert restored["citations"][0]["label"] == 1
    assert restored["contexts"][0]["chunk"]["chunk_id"] == "c0"
    assert restored["usage"]["provider"] == "fake"
    assert restored["insufficient_context"] is False


def test_document_and_chunk_to_dict_round_trip_metadata():
    doc = Document("d0", "docs/a.md", "t", "h", {"lang": "en", "n": 3})
    assert json.loads(json.dumps(doc.to_dict()))["metadata"] == {"lang": "en", "n": 3}
    chunk = make_chunk(metadata={"heading_path": "A/B"})
    assert chunk.to_dict()["metadata"] == {"heading_path": "A/B"}


# --- LoadIssue / IngestReport ------------------------------------------


def test_load_issue_is_frozen():
    issue = LoadIssue(source_uri="a.pdf", reason="unrecognized extension")
    with pytest.raises(dataclasses.FrozenInstanceError):
        issue.reason = "other"


@pytest.mark.parametrize("field_name", ["source_uri", "reason"])
def test_load_issue_rejects_empty_strings(field_name):
    kwargs = dict(source_uri="a.pdf", reason="skipped")
    kwargs[field_name] = ""
    with pytest.raises(ValueError):
        LoadIssue(**kwargs)


def test_ingest_report_rejects_non_tuple_fields():
    with pytest.raises(TypeError):
        IngestReport(loaded=[], skipped=(), failed=())


def test_ingest_report_to_dict_is_json_serialisable():
    doc = Document("d0", "docs/a.md", "text", "hash")
    report = IngestReport(
        loaded=(doc,),
        skipped=(LoadIssue("img.png", "unrecognized extension"),),
        failed=(LoadIssue("bad.txt", "contains a NUL byte"),),
    )
    restored = json.loads(json.dumps(report.to_dict()))
    assert restored["loaded"][0]["doc_id"] == "d0"
    assert restored["skipped"][0]["reason"] == "unrecognized extension"
    assert restored["failed"][0]["source_uri"] == "bad.txt"
