"""``ContextBuilder``: the C2 checkpoint (context provably never exceeds the
budget — property-tested with adversarial chunk sizes and non-additive
counters), dedup, labelling, the drop-lowest-ranked-first truncation policy,
and the ``Context`` / ``ContextBlock`` value types."""

import json
import warnings

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from nanorag.context import (
    BLOCK_SEPARATOR,
    Context,
    ContextBlock,
    ContextBuilder,
    render_block,
    render_context,
)
from nanorag.errors import ConfigError
from nanorag.hashing import chunk_id, normalize_text
from nanorag.tokens import HeuristicCounter, TiktokenCounter
from nanorag.types import Chunk, ScoredChunk

# --- helpers ------------------------------------------------------------------


def _hit(text, *, score=1.0, doc="doc", ordinal=0, cid=None) -> ScoredChunk:
    chunk = Chunk(
        chunk_id=cid or chunk_id(doc, ordinal, text),
        doc_id=doc,
        ordinal=ordinal,
        text=text,
        start_char=0,
        end_char=len(text),
        token_count=len(text.split()),
    )
    return ScoredChunk(chunk=chunk, score=score, source="dense")


class WordCounter:
    """Additive: one token per whitespace-separated word."""

    def count(self, text: str) -> int:
        return len(text.split())


class SeparatorHeavyCounter:
    """Super-additive: every block separator costs 5 extra tokens.

    Summing per-block counts under-estimates the joined text, so a builder
    that budgeted on the sum would exceed the budget.
    """

    def count(self, text: str) -> int:
        return len(text.split()) + 5 * text.count(BLOCK_SEPARATOR)


class DistinctCharCounter:
    """Sub-additive and order-independent: one token per distinct character."""

    def count(self, text: str) -> int:
        return len(set(text))


# --- construction -------------------------------------------------------------


@pytest.mark.parametrize("budget", [0, -1])
def test_non_positive_budget_is_a_config_error(budget):
    with pytest.raises(ConfigError):
        ContextBuilder(budget)


def test_default_counter_is_heuristic_and_does_not_warn():
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        builder = ContextBuilder(100)
    assert isinstance(builder.counter, HeuristicCounter)
    assert builder.budget_tokens == 100


# --- rendering ----------------------------------------------------------------


def test_render_block_is_label_line_then_verbatim_text():
    assert render_block(3, "  keep my   spacing \n") == "[3]\n  keep my   spacing \n"


def test_render_context_joins_with_separator_and_is_empty_for_no_blocks():
    a = ContextBlock(label=1, hit=_hit("a"), text="[1]\na", token_count=1)
    b = ContextBlock(label=2, hit=_hit("b"), text="[2]\nb", token_count=1)
    assert render_context([]) == ""
    assert render_context([a, b]) == "[1]\na" + BLOCK_SEPARATOR + "[2]\nb"


# --- the happy path -----------------------------------------------------------


def test_empty_input_gives_an_empty_untruncated_context():
    ctx = ContextBuilder(10, counter=WordCounter()).build([])
    assert ctx.blocks == ()
    assert ctx.text == ""
    assert ctx.token_count == 0
    assert ctx.truncated is False
    assert dict(ctx.labels) == {}
    assert ctx.hits == ()


def test_labels_follow_rank_order_and_map_back_to_chunk_ids():
    hits = [
        _hit("first", score=0.9),
        _hit("second", score=0.5),
        _hit("third", score=0.1),
    ]
    ctx = ContextBuilder(100, counter=WordCounter()).build(hits)
    assert [b.label for b in ctx.blocks] == [1, 2, 3]
    assert ctx.hits == tuple(hits)
    assert dict(ctx.labels) == {h.chunk.chunk_id: i + 1 for i, h in enumerate(hits)}
    assert ctx.text == "[1]\nfirst\n\n[2]\nsecond\n\n[3]\nthird"
    assert ctx.token_count == WordCounter().count(ctx.text)
    assert ctx.truncated is False


def test_input_order_is_preserved_not_resorted_by_score():
    # A reranker or fusion stage upstream owns the order; the builder must
    # not undo it by re-sorting on score.
    hits = [_hit("low", score=0.1), _hit("high", score=0.9)]
    ctx = ContextBuilder(100, counter=WordCounter()).build(hits)
    assert [b.hit.chunk.text for b in ctx.blocks] == ["low", "high"]


def test_per_block_token_count_is_the_rendered_block():
    ctx = ContextBuilder(100, counter=WordCounter()).build([_hit("two words")])
    # "[1]" is itself a word under this counter.
    assert ctx.blocks[0].token_count == 3
    assert ctx.blocks[0].text == "[1]\ntwo words"


# --- dedup --------------------------------------------------------------------


def test_duplicate_chunk_id_keeps_the_first_occurrence_only():
    hit = _hit("same chunk")
    ctx = ContextBuilder(100, counter=WordCounter()).build([hit, hit])
    assert len(ctx.blocks) == 1
    assert ctx.truncated is False


@pytest.mark.parametrize(
    "variant",
    [
        "para\r\ngraph",  # CRLF vs LF
        "para\ngraph  ",  # trailing whitespace
        "  para\ngraph",  # leading whitespace
        "para\ngraph",  # identical text, different document
    ],
)
def test_duplicate_normalised_text_across_documents_keeps_the_first(variant):
    first = _hit("para\ngraph", doc="a")
    second = _hit(variant, doc="b")
    assert first.chunk.chunk_id != second.chunk.chunk_id
    ctx = ContextBuilder(100, counter=WordCounter()).build([first, second])
    assert ctx.hits == (first,)
    assert ctx.truncated is False


def test_nfc_and_nfd_spellings_are_the_same_text():
    nfc = _hit("caf" + chr(0x00E9), doc="a")
    nfd = _hit("cafe" + chr(0x0301), doc="b")
    assert normalize_text(nfc.chunk.text) == normalize_text(nfd.chunk.text)
    ctx = ContextBuilder(100, counter=WordCounter()).build([nfc, nfd])
    assert ctx.hits == (nfc,)


@pytest.mark.parametrize("text", ["", "   ", "\n\n\t", "\r\n"])
def test_empty_or_whitespace_only_chunks_are_skipped_without_truncation(text):
    real = _hit("content", doc="b")
    ctx = ContextBuilder(100, counter=WordCounter()).build([_hit(text), real])
    assert ctx.hits == (real,)
    assert ctx.blocks[0].label == 1
    assert ctx.truncated is False


# --- truncation policy --------------------------------------------------------


def test_drops_lowest_ranked_first_and_reports_truncation():
    hits = [_hit("a b", score=0.9), _hit("c d", score=0.5), _hit("e f", score=0.1)]
    # Each block is 3 words ("[n]" + two); two blocks = 6, three = 9.
    ctx = ContextBuilder(6, counter=WordCounter()).build(hits)
    assert ctx.hits == (hits[0], hits[1])
    assert ctx.truncated is True
    assert ctx.token_count == 6


def test_stops_at_the_first_block_that_does_not_fit_leaving_no_holes():
    # Rank 2 is too big; rank 3 would fit on its own. The context must be
    # a rank-order prefix, so rank 3 is not pulled forward past rank 2.
    hits = [_hit("a"), _hit(" ".join(["big"] * 20)), _hit("c")]
    ctx = ContextBuilder(6, counter=WordCounter()).build(hits)
    assert ctx.hits == (hits[0],)
    assert ctx.truncated is True


def test_a_single_chunk_larger_than_the_budget_gives_zero_blocks_truncated():
    ctx = ContextBuilder(3, counter=WordCounter()).build([_hit("one two three")])
    assert ctx.blocks == ()
    assert ctx.text == ""
    assert ctx.token_count == 0
    assert ctx.truncated is True


def test_exact_fit_is_included_and_one_token_less_is_not():
    hit = _hit("two words")  # rendered: 3 words
    assert ContextBuilder(3, counter=WordCounter()).build([hit]).hits == (hit,)
    assert ContextBuilder(2, counter=WordCounter()).build([hit]).hits == ()


def test_a_chunk_is_never_split_to_fit():
    hit = _hit("alpha beta gamma delta")
    ctx = ContextBuilder(4, counter=WordCounter()).build([hit])
    assert ctx.blocks == ()  # dropped whole, not trimmed to "alpha beta gamma"


def test_budget_is_checked_on_the_joined_text_not_a_per_block_sum():
    # Two one-word blocks: 2 words each rendered ("[n]" + word) -> sum 4.
    # The separator between them costs 5 under this counter -> joined 9.
    hits = [_hit("a"), _hit("b")]
    counter = SeparatorHeavyCounter()
    ctx = ContextBuilder(8, counter=counter).build(hits)
    assert sum(b.token_count for b in ctx.blocks) <= 8
    assert ctx.hits == (hits[0],)
    assert ctx.truncated is True
    assert counter.count(ctx.text) <= 8


def test_real_tiktoken_counter_respects_the_budget():
    counter = TiktokenCounter("cl100k_base")
    hits = [
        _hit(" ".join(f"w{n}_{i}" for i in range(40)), doc=f"d{n}") for n in range(6)
    ]
    # A budget that fits the first two rendered blocks exactly, under the
    # real tokenizer, and therefore cannot fit all six.
    budget = counter.count(ContextBuilder(10**6, counter=counter).build(hits[:2]).text)
    ctx = ContextBuilder(budget, counter=counter).build(hits)
    assert ctx.hits == tuple(hits[:2])
    assert ctx.truncated is True
    assert counter.count(ctx.text) <= budget
    assert ctx.token_count == counter.count(ctx.text)


def test_build_is_deterministic():
    hits = [_hit("a b c", doc="x"), _hit("d e", doc="y"), _hit("f", doc="z")]
    builder = ContextBuilder(7, counter=WordCounter())
    assert builder.build(hits) == builder.build(hits)


# --- the value types ----------------------------------------------------------


def test_context_refuses_a_token_count_over_budget():
    with pytest.raises(ValueError):
        Context(
            blocks=(),
            labels={},
            text="",
            token_count=5,
            budget_tokens=4,
            truncated=False,
        )


def test_context_blocks_must_be_a_tuple():
    with pytest.raises(TypeError):
        Context(
            blocks=[],
            labels={},
            text="",
            token_count=0,
            budget_tokens=1,
            truncated=False,
        )


def test_context_labels_are_read_only():
    ctx = ContextBuilder(100, counter=WordCounter()).build([_hit("a")])
    with pytest.raises(TypeError):
        ctx.labels["x"] = 9  # type: ignore[index]


@pytest.mark.parametrize("kwargs", [{"label": 0}, {"token_count": -1}])
def test_context_block_validation(kwargs):
    base = {"label": 1, "hit": _hit("a"), "text": "[1]\na", "token_count": 1}
    with pytest.raises(ValueError):
        ContextBlock(**{**base, **kwargs})


def test_context_to_dict_is_json_serialisable():
    ctx = ContextBuilder(100, counter=WordCounter()).build([_hit("a"), _hit("b")])
    payload = json.loads(json.dumps(ctx.to_dict()))
    assert payload["truncated"] is False
    assert payload["budget_tokens"] == 100
    assert payload["labels"] == {
        h.chunk.chunk_id: i + 1 for i, h in enumerate(ctx.hits)
    }
    assert [b["label"] for b in payload["blocks"]] == [1, 2]
    assert payload["blocks"][0]["hit"]["chunk"]["text"] == "a"


# --- the C2 checkpoint: context provably never exceeds budget ------------------

_COUNTERS = [
    HeuristicCounter(warn=False),
    WordCounter(),
    SeparatorHeavyCounter(),
    DistinctCharCounter(),
]

# Adversarial sizes: empty, whitespace-only, a single character, short, and
# far larger than any budget below — plus a tiny alphabet so texts collide
# and the dedup path is exercised, not just the budget path.
_TEXT = st.one_of(
    st.just(""),
    st.sampled_from([" ", "\n", " \r\n\t "]),
    st.text(alphabet="ab \n", min_size=1, max_size=12),
    st.text(alphabet="ab ", min_size=150, max_size=300),
)


@st.composite
def _ranked_hits(draw):
    texts = draw(st.lists(_TEXT, max_size=8))
    hits = [_hit(t, doc=f"d{i}", score=1.0 - i / 10) for i, t in enumerate(texts)]
    # Re-insert some hits (same chunk id) and some copies of a text under a
    # new document (same text, new id) at random positions.
    for _ in range(draw(st.integers(min_value=0, max_value=3))):
        if not hits:
            break
        source = hits[draw(st.integers(min_value=0, max_value=len(hits) - 1))]
        if draw(st.booleans()):
            dup = source
        else:
            dup = _hit(source.chunk.text, doc=f"copy{len(hits)}", score=0.05)
        hits.insert(draw(st.integers(min_value=0, max_value=len(hits))), dup)
    return hits


def _reference_candidates(hits):
    """The dedup rule, restated independently of the implementation."""
    seen_ids, seen_texts, kept = set(), set(), []
    for hit in hits:
        key = normalize_text(hit.chunk.text).strip()
        if key and hit.chunk.chunk_id not in seen_ids and key not in seen_texts:
            seen_ids.add(hit.chunk.chunk_id)
            seen_texts.add(key)
            kept.append(hit)
    return kept


@settings(max_examples=500, deadline=None)
@given(
    hits=_ranked_hits(),
    budget=st.integers(min_value=1, max_value=60),
    counter=st.sampled_from(_COUNTERS),
)
def test_context_never_exceeds_budget_and_is_a_rank_order_prefix(hits, budget, counter):
    ctx = ContextBuilder(budget, counter=counter).build(hits)

    # The checkpoint: measured on the joined text with the real counter.
    assert counter.count(ctx.text) <= budget
    assert ctx.token_count == counter.count(ctx.text)
    assert ctx.text == render_context(ctx.blocks)

    # Blocks are a prefix of the deduplicated candidates, in order.
    candidates = _reference_candidates(hits)
    assert list(ctx.hits) == candidates[: len(ctx.blocks)]

    # Labels are 1..n in block order and the map is exactly those blocks.
    assert [b.label for b in ctx.blocks] == list(range(1, len(ctx.blocks) + 1))
    assert dict(ctx.labels) == {b.hit.chunk.chunk_id: b.label for b in ctx.blocks}
    assert all(b.text == render_block(b.label, b.hit.chunk.text) for b in ctx.blocks)

    # Truncated iff something was dropped for budget — and then the very
    # next candidate genuinely would not have fit (greedy is maximal).
    assert ctx.truncated == (len(ctx.blocks) < len(candidates))
    if ctx.truncated:
        nxt = candidates[len(ctx.blocks)]
        extra = ContextBlock(
            label=len(ctx.blocks) + 1,
            hit=nxt,
            text=render_block(len(ctx.blocks) + 1, nxt.chunk.text),
            token_count=0,
        )
        assert counter.count(render_context([*ctx.blocks, extra])) > budget
