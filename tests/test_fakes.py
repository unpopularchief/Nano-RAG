import numpy as np
import pytest

from nanorag.errors import QuotaExhausted
from nanorag.generation.base import Generation, Generator
from nanorag.prompting.templates import Prompt
from tests.fakes import FakeClock, FakeEmbedder, FakeGenerator


def cosine(a: np.ndarray, b: np.ndarray) -> float:
    return float(a @ b)  # both are unit vectors


# --- FakeEmbedder ---------------------------------------------------------


def test_embedder_is_deterministic():
    e = FakeEmbedder()
    assert np.array_equal(e.embed_query("hello world"), e.embed_query("hello world"))


def test_embed_shapes_and_dtype():
    e = FakeEmbedder(dim=32)
    out = e.embed(["one", "two", "three"])
    assert out.shape == (3, 32)
    assert out.dtype == np.float32
    assert e.embed([]).shape == (0, 32)
    assert e.embed_query("x").shape == (32,)


def test_non_empty_text_gives_a_unit_vector():
    v = FakeEmbedder().embed_query("some words here")
    assert np.isclose(np.linalg.norm(v), 1.0)


def test_shared_vocabulary_raises_cosine_similarity():
    e = FakeEmbedder(dim=256)
    query = e.embed_query("cosine similarity retrieval")
    close = e.embed_query("retrieval uses cosine similarity")
    far = e.embed_query("penguins waddle across antarctic ice")
    assert cosine(query, close) > cosine(query, far)


def test_normalisation_makes_line_endings_irrelevant():
    e = FakeEmbedder()
    assert np.array_equal(e.embed_query("a b\r\nc"), e.embed_query("a b\nc"))


def test_embedder_rejects_bad_dim():
    with pytest.raises(ValueError):
        FakeEmbedder(dim=0)


# --- FakeGenerator ------------------------------------------------------


def _prompt(user: str) -> Prompt:
    return Prompt(system="sys", user=user, nonce="n")


def test_generator_records_prompts():
    g = FakeGenerator()
    g.generate(_prompt("first prompt"))
    g.generate(_prompt("second prompt"))
    assert [c.user for c in g.calls] == ["first prompt", "second prompt"]
    assert g.last_prompt == "sys" + chr(10) * 2 + "second prompt"


def test_generator_returns_scripted_then_default():
    g = FakeGenerator(["a", "b"], default="fallback")
    assert g.generate(_prompt("p")).text == "a"
    assert g.generate(_prompt("p")).text == "b"
    assert g.generate(_prompt("p")).text == "fallback"


def test_generator_queue_appends_responses():
    g = FakeGenerator(default="d")
    g.queue("x", "y")
    got = [g.generate(_prompt("p")).text for _ in range(3)]
    assert got == ["x", "y", "d"]


def test_generator_raises_a_scripted_exception_then_continues():
    g = FakeGenerator([QuotaExhausted("spent"), "after"])
    with pytest.raises(QuotaExhausted):
        g.generate(_prompt("p"))
    assert g.generate(_prompt("p")).text == "after"
    assert len(g.calls) == 2


def test_generator_reports_usage_and_satisfies_the_protocol():
    g = FakeGenerator(provider="fake", model="fake-generator")
    assert isinstance(g, Generator)
    gen = g.generate(_prompt("hello"))
    assert isinstance(gen, Generation)
    assert gen.usage.provider == "fake"
    assert gen.usage.model == "fake-generator"
    assert gen.usage.prompt_tokens > 0
    usage = gen.usage
    assert usage.total_tokens == usage.prompt_tokens + usage.completion_tokens


def test_generator_last_prompt_is_none_before_any_call():
    assert FakeGenerator().last_prompt is None


# --- FakeClock --------------------------------------------------------


def test_clock_sleep_advances_and_records():
    clock = FakeClock(start=100.0)
    clock.sleep(1.5)
    clock.sleep(0.5)
    assert clock.now() == 102.0
    assert clock.monotonic() == 102.0
    assert clock.sleeps == [1.5, 0.5]


def test_clock_advance_does_not_record_a_sleep():
    clock = FakeClock()
    clock.advance(10.0)
    assert clock.now() == 10.0
    assert clock.sleeps == []


def test_clock_rejects_negative_durations():
    clock = FakeClock()
    with pytest.raises(ValueError):
        clock.sleep(-1.0)
    with pytest.raises(ValueError):
        clock.advance(-1.0)
