"""``pipeline``: the full ingest -> answer path with fakes and no network
(plan.md §9 Phase C tests), zero retrieved chunks -> the defined
insufficient-context answer with no model call, truncation propagation, the
budget arithmetic, the F10 relative signal, ``Rag.query()`` reproduced from
public functions alone (plan.md §15 #5), persistence across a reopen, and
``default_generator`` / ``from_defaults`` wiring."""

import logging
import sqlite3

import httpx
import pytest

from nanorag import Rag
from nanorag.chunking import RecursiveChunker
from nanorag.config import Settings
from nanorag.context import ContextBuilder
from nanorag.errors import (
    ConfigError,
    IndexModelMismatch,
    RerankError,
    RetrievalError,
    StoreError,
)
from nanorag.generation import FallbackGenerator, Generation
from nanorag.hashing import content_hash, stable_doc_id
from nanorag.pipeline import (
    DEDUP_GROUP_KEY,
    DEDUP_OF_KEY,
    DEFAULT_MIN_SCORE,
    NO_PROVIDER,
    context_budget,
    default_generator,
    insufficient_context,
    load_vectors,
)
from nanorag.prompting import INSUFFICIENT_CONTEXT_TEXT, PromptBuilder
from nanorag.rerank.identity import IdentityReranker
from nanorag.retrieval import DenseRetriever
from nanorag.store import SqliteDocumentStore
from nanorag.tokens import HeuristicCounter
from nanorag.types import Answer, Chunk, Document, ScoredChunk, Usage
from tests.fakes import FakeEmbedder, FakeGenerator, FakeReranker

DIM = 256  # wide enough that the hashing fake has no accidental collisions

CORPUS = {
    "cats.txt": "Cats sleep most of the day. A cat purrs when content.",
    "dogs.txt": "Dogs bark at strangers. A dog wags its tail when happy.",
    "notes/fish.md": "# Fish\n\nFish swim in water. Goldfish live in bowls.",
    "image.png": b"\x89PNG not text",
}


def _write_corpus(root):
    for name, body in CORPUS.items():
        path = root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        if isinstance(body, bytes):
            path.write_bytes(body)
        else:
            path.write_text(body, encoding="utf-8")
    return root


def _doc(uri, text, metadata=None):
    return Document(
        doc_id=stable_doc_id(uri),
        source_uri=uri,
        text=text,
        content_hash=content_hash(text),
        metadata=metadata or {},
    )


def _rag(generator=None, docs=None, **kwargs):
    kwargs.setdefault("counter", HeuristicCounter(warn=False))
    return Rag(
        embedder=FakeEmbedder(dim=DIM),
        generator=generator if generator is not None else FakeGenerator(),
        docs=docs if docs is not None else SqliteDocumentStore(":memory:"),
        **kwargs,
    )


# --- the public functions -----------------------------------------------------


def test_context_budget_arithmetic_and_floor():
    assert context_budget(1000, 200, 100, margin=0.0) == 700
    assert context_budget(1000, 200, 100, margin=0.15) == int(700 * 0.85)
    assert context_budget(100, 90, 50) == 1  # never below 1
    with pytest.raises(ConfigError):
        context_budget(1000, 1, 1, margin=1.0)
    with pytest.raises(ConfigError):
        context_budget(1000, 1, 1, margin=-0.1)


def _hits(*scores):
    doc = _doc("d", "text")
    return [
        ScoredChunk(
            chunk=Chunk(
                chunk_id=f"c{i}",
                doc_id=doc.doc_id,
                ordinal=i,
                text="t",
                start_char=0,
                end_char=1,
                token_count=1,
            ),
            score=s,
            source="dense",
        )
        for i, s in enumerate(scores)
    ]


def test_insufficient_context_count_floor_and_relative_gap():
    assert insufficient_context([]) is True
    assert insufficient_context(_hits(0.9)) is False
    assert insufficient_context(_hits(0.9), min_results=2) is True
    # Gap disabled by default: a flat ranking still counts as sufficient.
    assert insufficient_context(_hits(0.50, 0.49, 0.49)) is False
    # Gap enabled: top-1 barely above the mean of the rest -> insufficient.
    assert insufficient_context(_hits(0.50, 0.49, 0.49), min_gap=0.05) is True
    assert insufficient_context(_hits(0.80, 0.49, 0.49), min_gap=0.05) is False
    # A single hit cannot fail the gap check.
    assert insufficient_context(_hits(0.10), min_gap=0.5) is False


def test_insufficient_context_absolute_floor():
    # Off by default: the score scale belongs to the embedder.
    assert insufficient_context(_hits(0.10)) is False
    # On: judged on the top hit only — the rest may be anything.
    assert insufficient_context(_hits(0.54, 0.10), min_score=0.55) is True
    assert insufficient_context(_hits(0.55, 0.10), min_score=0.55) is False
    # Independent of the other two checks: a clear gap does not rescue a
    # top hit under the floor, and a floor-clearing hit still needs the gap.
    assert insufficient_context(_hits(0.50, 0.10), min_score=0.55, min_gap=0.1) is True
    assert insufficient_context(_hits(0.70, 0.69), min_score=0.55, min_gap=0.1) is True
    assert insufficient_context(_hits(0.70, 0.10), min_score=0.55, min_gap=0.1) is False
    # The count floor comes first: no hits is insufficient whatever the floor.
    assert insufficient_context([], min_score=0.0) is True


def test_load_vectors_rebuilds_the_index_from_sqlite():
    docs = SqliteDocumentStore(":memory:")
    assert len(load_vectors(docs, 4)) == 0
    rag = _rag(docs=docs)
    rag.ingest([_doc("a.txt", "alpha beta"), _doc("b.txt", "gamma delta")])
    rebuilt = load_vectors(docs, DIM)
    assert len(rebuilt) == len(rag.vectors)
    q = rag.embedder.embed_query("alpha")
    assert rebuilt.search(q, 5) == rag.vectors.search(q, 5)


# --- construction -------------------------------------------------------------


def test_index_built_with_another_model_is_refused():
    docs = SqliteDocumentStore(":memory:")
    _rag(docs=docs).ingest([_doc("a.txt", "alpha")])
    with pytest.raises(IndexModelMismatch):
        Rag(
            embedder=FakeEmbedder(dim=DIM, model_id="other-model"),
            generator=FakeGenerator(),
            docs=docs,
        )
    with pytest.raises(IndexModelMismatch):
        Rag(embedder=FakeEmbedder(dim=16), generator=FakeGenerator(), docs=docs)


def test_defaults_are_wired_and_overridable():
    rag = _rag(k=3)
    assert rag.retriever.k == 3
    assert isinstance(rag.prompt_builder, PromptBuilder)
    assert rag.chunker.__class__.__name__ == "RecursiveChunker"
    assert rag.loader.__class__.__name__ == "DirectoryLoader"


def test_close_closes_docs_and_a_closable_generator():
    class Closable(FakeGenerator):
        closed = False

        def close(self):
            self.closed = True

    gen = Closable()
    rag = _rag(generator=gen)
    rag.close()
    assert gen.closed
    with pytest.raises(sqlite3.ProgrammingError):
        rag.docs.iter_documents().__next__()


# --- write path ---------------------------------------------------------------


def test_ingest_path_loads_chunks_embeds_and_indexes(tmp_path):
    rag = _rag()
    report = rag.ingest_path(_write_corpus(tmp_path))
    assert sorted(d.source_uri for d in report.loaded) == [
        "cats.txt",
        "dogs.txt",
        "notes/fish.md",
    ]
    assert [i.source_uri for i in report.skipped] == ["image.png"]
    assert report.failed == ()
    assert len(rag.vectors) == sum(1 for _ in rag.docs.iter_chunks()) == 3
    assert rag.docs.index_meta() == ("fake-embedder", DIM)


def test_reingest_of_a_changed_document_drops_its_stale_vectors():
    rag = _rag()
    rag.ingest([_doc("a.txt", "alpha beta gamma")])
    old_ids = {c.chunk_id for c in rag.docs.get_chunks(stable_doc_id("a.txt"))}
    rag.ingest([_doc("a.txt", "completely different words")])
    new_ids = {c.chunk_id for c in rag.docs.get_chunks(stable_doc_id("a.txt"))}
    assert old_ids.isdisjoint(new_ids)
    assert len(rag.vectors) == len(new_ids) == 1
    hits = rag.retrieve("alpha beta gamma", k=5)
    assert {h.chunk.chunk_id for h in hits} == new_ids


def test_reingest_of_an_unchanged_document_is_idempotent():
    rag = _rag()
    rag.ingest([_doc("a.txt", "alpha beta")])
    rag.ingest([_doc("a.txt", "alpha beta")])
    assert len(rag.vectors) == 1
    assert rag.ingest([_doc("empty.txt", "")]) == 0
    assert rag.docs.get_document(stable_doc_id("empty.txt")) is not None


class _CountingChunker:
    """Wraps a real chunker, counting calls (change-detection spy)."""

    def __init__(self, inner):
        self.inner = inner
        self.calls = 0

    def chunk(self, document):
        self.calls += 1
        return self.inner.chunk(document)


class _CountingEmbedder(FakeEmbedder):
    """A ``FakeEmbedder`` that counts its own ``embed`` calls."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.calls = 0

    def embed(self, texts):
        self.calls += 1
        return super().embed(texts)


def test_reingest_of_an_unchanged_document_skips_rechunking_and_reembedding():
    chunker = _CountingChunker(RecursiveChunker())
    embedder = _CountingEmbedder(dim=DIM)
    rag = Rag(
        embedder=embedder,
        generator=FakeGenerator(),
        docs=SqliteDocumentStore(":memory:"),
        chunker=chunker,
        counter=HeuristicCounter(warn=False),
    )
    total_first = rag.ingest([_doc("a.txt", "alpha beta gamma")])
    assert chunker.calls == 1
    assert embedder.calls == 1

    total_second = rag.ingest([_doc("a.txt", "alpha beta gamma")])
    assert chunker.calls == 1  # unchanged content_hash -> no rechunk
    assert embedder.calls == 1  # -> no re-embed either
    assert total_second == total_first == 1
    assert len(rag.vectors) == 1

    rag.ingest([_doc("a.txt", "totally different text now")])
    assert chunker.calls == 2  # a real change still triggers a full rechunk


def test_delete_document_removes_it_from_search_and_sqlite():
    rag = _rag()
    rag.ingest([_doc("a.txt", "alpha beta gamma"), _doc("b.txt", "delta epsilon")])
    doc_id = stable_doc_id("a.txt")

    removed = rag.delete_document(doc_id)

    assert removed == 1
    assert rag.docs.get_document(doc_id) is None
    assert rag.docs.get_chunks(doc_id) == []
    assert rag.docs.count_chunks() == 1  # only b.txt's chunk remains
    hits = rag.retrieve("alpha beta gamma", k=5)
    assert doc_id not in {h.chunk.doc_id for h in hits}
    assert len(rag.vectors) == 1


def test_delete_document_unknown_doc_id_is_a_noop():
    rag = _rag()
    rag.ingest([_doc("a.txt", "alpha beta gamma")])
    assert rag.delete_document("does-not-exist") == 0
    assert len(rag.vectors) == 1


def test_compact_after_delete_leaves_search_unchanged():
    rag = _rag()
    rag.ingest([_doc("a.txt", "alpha beta gamma"), _doc("b.txt", "delta epsilon")])
    rag.delete_document(stable_doc_id("a.txt"))
    before = rag.retrieve("delta epsilon", k=5)

    rag.compact()

    after = rag.retrieve("delta epsilon", k=5)
    assert [h.chunk.chunk_id for h in before] == [h.chunk.chunk_id for h in after]
    assert len(rag.vectors) == 1

    # The reclaimed row is reused cleanly by a later ingest (plan.md §6).
    rag.ingest([_doc("c.txt", "zeta eta theta")])
    assert len(rag.vectors) == 2


# --- sync_path -----------------------------------------------------------------


def test_sync_path_dry_run_reports_the_plan_without_writing_anything(tmp_path):
    rag = _rag()
    (tmp_path / "a.txt").write_text("alpha beta", encoding="utf-8")
    (tmp_path / "b.txt").write_text("delta epsilon", encoding="utf-8")
    rag.sync_path(tmp_path, apply=True)

    (tmp_path / "b.txt").write_text("changed content entirely", encoding="utf-8")
    (tmp_path / "a.txt").unlink()
    (tmp_path / "c.txt").write_text("zeta eta theta", encoding="utf-8")

    report = rag.sync_path(tmp_path)

    assert report.added == ("c.txt",)
    assert report.updated == ("b.txt",)
    assert report.unchanged == ()
    assert report.deleted == ("a.txt",)
    assert report.applied is False
    # Nothing was actually written: the store still matches the first sync.
    assert {d.source_uri for d in rag.docs.iter_documents()} == {"a.txt", "b.txt"}
    assert rag.docs.get_document(stable_doc_id("b.txt")).text == "delta epsilon"


def test_sync_path_apply_executes_add_update_and_delete():
    rag = _rag()
    rag.ingest([_doc("a.txt", "alpha beta"), _doc("b.txt", "delta epsilon")])

    class _FakeLoader:
        def load_path(self, root, *, glob, ignore):
            from nanorag.types import IngestReport

            return IngestReport(
                loaded=(
                    _doc("b.txt", "changed content entirely"),
                    _doc("c.txt", "zeta eta theta"),
                ),
                skipped=(),
                failed=(),
            )

    rag.loader = _FakeLoader()
    report = rag.sync_path("ignored")

    assert report.added == ("c.txt",)
    assert report.updated == ("b.txt",)
    assert report.deleted == ("a.txt",)
    assert report.applied is False

    applied = rag.sync_path("ignored", apply=True)
    assert applied.applied is True
    assert {d.source_uri for d in rag.docs.iter_documents()} == {"b.txt", "c.txt"}
    assert rag.docs.get_document(stable_doc_id("b.txt")).text == (
        "changed content entirely"
    )
    hits = rag.retrieve("alpha beta", k=5)
    assert stable_doc_id("a.txt") not in {h.chunk.doc_id for h in hits}


def test_sync_path_apply_refuses_past_max_delete_fraction(tmp_path):
    rag = _rag()
    for name in ("a.txt", "b.txt", "c.txt", "d.txt"):
        (tmp_path / name).write_text(name, encoding="utf-8")
    rag.sync_path(tmp_path, apply=True)
    assert rag.docs.count_documents() == 4

    for name in ("a.txt", "b.txt", "c.txt"):  # 3/4 = 75% > the 50% default
        (tmp_path / name).unlink()

    dry = rag.sync_path(tmp_path)
    assert dry.over_delete_guard is True
    assert dry.applied is False

    with pytest.raises(StoreError):
        rag.sync_path(tmp_path, apply=True)
    # A blocked sync writes nothing at all -- not even the safe adds/updates.
    assert rag.docs.count_documents() == 4

    applied = rag.sync_path(tmp_path, apply=True, max_delete_fraction=0.9)
    assert applied.applied is True
    assert rag.docs.count_documents() == 1


def test_sync_path_rejects_an_out_of_range_max_delete_fraction(tmp_path):
    rag = _rag()
    with pytest.raises(ConfigError):
        rag.sync_path(tmp_path, max_delete_fraction=1.5)
    with pytest.raises(ConfigError):
        rag.sync_path(tmp_path, max_delete_fraction=-0.1)


# --- near-duplicate detection (plan.md §18 F13) ---------------------------------


def test_near_duplicate_chunk_is_kept_and_tagged_with_the_winner():
    rag = _rag()
    rag.ingest([_doc("a.txt", "alpha beta gamma", {DEDUP_GROUP_KEY: "g"})])
    winner_id = rag.docs.get_chunks(stable_doc_id("a.txt"))[0].chunk_id

    rag.ingest([_doc("b.txt", "alpha beta gamma", {DEDUP_GROUP_KEY: "g"})])

    loser = rag.docs.get_chunks(stable_doc_id("b.txt"))[0]
    assert loser.metadata[DEDUP_OF_KEY] == winner_id
    # The loser is kept, not dropped: still searchable, still in SQLite.
    assert len(rag.vectors) == 2
    assert rag.docs.count_chunks() == 2


def test_near_duplicate_detection_is_a_noop_without_a_dedup_group():
    rag = _rag()
    rag.ingest([_doc("a.txt", "alpha beta gamma")])
    rag.ingest([_doc("b.txt", "alpha beta gamma")])

    for chunk in rag.docs.get_chunks(stable_doc_id("b.txt")):
        assert DEDUP_OF_KEY not in chunk.metadata


def test_tag_near_duplicates_exclude_keeps_a_chunk_from_matching_its_own_document():
    # _tag_near_duplicates is called mid-ingest with `exclude` set to this
    # document's own current and prior chunk ids (plan.md §18 F13: a chunk
    # is never compared against its own document). Exercised directly here
    # because a same-document match can only arise via a same-content
    # re-ingest, which is awkward to stage through the public API alone.
    rag = _rag()
    rag.ingest([_doc("seed.txt", "alpha beta gamma", {DEDUP_GROUP_KEY: "g"})])
    seed_id = rag.docs.get_chunks(stable_doc_id("seed.txt"))[0].chunk_id

    doc = _doc("a.txt", "alpha beta gamma", {DEDUP_GROUP_KEY: "g"})
    chunk = Chunk(
        chunk_id="a-chunk",
        doc_id=doc.doc_id,
        ordinal=0,
        text="alpha beta gamma",
        start_char=0,
        end_char=16,
        token_count=3,
    )
    vector = rag.embedder.embed(["alpha beta gamma"])

    excluded = rag._tag_near_duplicates(doc, [chunk], vector, exclude={seed_id})
    assert DEDUP_OF_KEY not in excluded[0].metadata

    included = rag._tag_near_duplicates(doc, [chunk], vector, exclude=set())
    assert included[0].metadata[DEDUP_OF_KEY] == seed_id


def test_near_duplicate_below_threshold_is_not_flagged():
    rag = _rag(duplicate_threshold=0.999)
    rag.ingest([_doc("a.txt", "alpha beta gamma", {DEDUP_GROUP_KEY: "g"})])
    rag.ingest([_doc("b.txt", "alpha beta gamma delta", {DEDUP_GROUP_KEY: "g"})])

    for chunk in rag.docs.get_chunks(stable_doc_id("b.txt")):
        assert DEDUP_OF_KEY not in chunk.metadata


# --- Gate E review: literal checks for the plan.md §9 Phase E Tests/Acceptance
# bullets that E1/E2 exercised only indirectly ------------------------------


def test_editing_one_document_leaves_others_ids_and_ordinals_untouched():
    rag = _rag()
    rag.ingest([_doc("a.txt", "alpha beta gamma"), _doc("b.txt", "delta epsilon zeta")])
    b_before = rag.docs.get_chunks(stable_doc_id("b.txt"))

    rag.ingest([_doc("a.txt", "a completely rewritten document now")])

    b_after = rag.docs.get_chunks(stable_doc_id("b.txt"))
    assert b_after == b_before  # same ids, same ordinals, same text -- untouched


def test_full_add_edit_delete_cycle_leaves_no_orphan_rows(tmp_path):
    rag = _rag()
    for name in ("a.txt", "b.txt", "c.txt"):
        (tmp_path / name).write_text(name, encoding="utf-8")
    rag.sync_path(tmp_path, apply=True)

    (tmp_path / "b.txt").write_text("b edited", encoding="utf-8")  # update
    (tmp_path / "c.txt").unlink()  # delete
    (tmp_path / "d.txt").write_text("d.txt", encoding="utf-8")  # add
    report = rag.sync_path(tmp_path, apply=True)
    assert report.applied is True

    conn = rag.docs._conn
    doc_ids = {row[0] for row in conn.execute("SELECT doc_id FROM documents")}
    chunk_rows = conn.execute("SELECT chunk_id, doc_id FROM chunks").fetchall()
    chunk_ids = {row[0] for row in chunk_rows}
    embedding_chunk_ids = {
        row[0] for row in conn.execute("SELECT chunk_id FROM embeddings")
    }
    # No chunk row points at a document that no longer exists, and no
    # embedding row points at a chunk that no longer exists -- the literal
    # "no orphan rows in any table" acceptance line (plan.md §9 Phase E).
    assert {doc_id for _, doc_id in chunk_rows} <= doc_ids
    assert embedding_chunk_ids == chunk_ids


def test_interrupted_ingest_leaves_no_orphan_vectors_after_reload():
    rag = _rag()
    rag.ingest([_doc("a.txt", "alpha beta gamma")])
    old_chunk_id = rag.docs.get_chunks(stable_doc_id("a.txt"))[0].chunk_id

    def _crash(chunk_ids):
        raise RuntimeError(
            "simulated crash between the SQLite commit and the "
            "in-memory index update (plan.md §18 F2)"
        )

    rag.vectors.delete = _crash
    with pytest.raises(RuntimeError):
        rag.ingest([_doc("a.txt", "a completely rewritten document now")])

    # The SQLite write already committed by the time the crash hits -- only
    # the chunk id changed there, exactly as an uninterrupted ingest would.
    new_chunk_id = rag.docs.get_chunks(stable_doc_id("a.txt"))[0].chunk_id
    assert new_chunk_id != old_chunk_id

    # The crashed process's in-memory index is now stale (still holds the
    # old id, is missing the new one) -- but a fresh rebuild from SQLite
    # alone (what a restart does, plan.md §18 F2) is exact: no orphans.
    rebuilt = load_vectors(rag.docs, dim=DIM)
    assert set(rebuilt._ids) == {new_chunk_id}


# --- read path: the e2e with fakes --------------------------------------------


def test_ingest_to_answer_end_to_end_with_fakes(tmp_path):
    generator = FakeGenerator(["Cats sleep most of the day. [1]"])
    rag = _rag(generator=generator, k=2)
    rag.ingest_path(_write_corpus(tmp_path))

    answer = rag.query("cats sleep purrs")

    assert isinstance(answer, Answer)
    assert answer.text == "Cats sleep most of the day. [1]"
    assert answer.insufficient_context is False
    assert answer.truncated is False
    assert answer.usage.provider == "fake"
    assert answer.usage.prompt_tokens > 0

    # The best hit is the cats chunk, and [1] resolves to it.
    assert answer.contexts[0].chunk.doc_id == stable_doc_id("cats.txt")
    assert len(answer.contexts) == 2
    assert len(answer.citations) == 1
    citation = answer.citations[0]
    assert citation.label == 1
    assert citation.chunk_id == answer.contexts[0].chunk.chunk_id
    assert citation.doc_id == stable_doc_id("cats.txt")
    assert citation.source_uri == "cats.txt"
    assert (citation.start_char, citation.end_char) == (
        answer.contexts[0].chunk.start_char,
        answer.contexts[0].chunk.end_char,
    )
    # Exactly those chunks were in the prompt, fenced, in label order.
    prompt = generator.calls[0]
    assert prompt.user.index("[1]\n" + answer.contexts[0].chunk.text) < (
        prompt.user.index("[2]\n" + answer.contexts[1].chunk.text)
    )
    assert "=== BEGIN UNTRUSTED CONTEXT " in prompt.user
    assert prompt.user.endswith("Question: cats sleep purrs")
    assert prompt.nonce in prompt.system

    t = answer.timings
    assert t.retrieve_ms >= 0 and t.context_ms >= 0 and t.generate_ms >= 0
    assert t.total_ms >= t.retrieve_ms + t.context_ms + t.generate_ms
    assert t.embed_ms == t.rerank_ms == 0.0


def test_zero_retrieved_chunks_gives_the_defined_answer_and_no_model_call():
    generator = FakeGenerator()
    rag = _rag(generator=generator)  # empty index
    answer = rag.query("Anything?")
    assert answer.text == INSUFFICIENT_CONTEXT_TEXT
    assert answer.insufficient_context is True
    assert answer.truncated is False
    assert answer.contexts == ()
    assert answer.usage == Usage(NO_PROVIDER, "", 0, 0, 0)
    assert generator.calls == []
    assert answer.timings.generate_ms == 0.0


def test_filter_matching_nothing_also_abstains_without_a_model_call():
    generator = FakeGenerator()
    rag = _rag(generator=generator)
    rag.ingest([_doc("a.txt", "alpha beta")])
    answer = rag.query("alpha", filter={"source_uri": "nope.txt"})
    assert answer.text == INSUFFICIENT_CONTEXT_TEXT
    assert answer.insufficient_context is True
    assert generator.calls == []


def test_context_that_fits_nothing_abstains_and_reports_truncation():
    # A generator whose window barely covers the system prompt: budget 1.
    generator = FakeGenerator(context_window=300, max_output_tokens=10)
    rag = _rag(generator=generator)
    rag.ingest([_doc("a.txt", "alpha " * 50)])
    answer = rag.query("alpha")
    assert answer.text == INSUFFICIENT_CONTEXT_TEXT
    assert answer.insufficient_context is True
    assert answer.truncated is True
    assert generator.calls == []


def test_partial_fit_calls_the_model_and_flags_truncation():
    # Room for roughly one chunk, not three.
    generator = FakeGenerator(context_window=420, max_output_tokens=10)
    rag = _rag(generator=generator, budget_margin=0.0)
    rag.ingest([_doc(f"{i}.txt", f"word{i} " * 30) for i in range(3)])
    answer = rag.query("word0 word1 word2", k=3)
    assert answer.truncated is True
    assert answer.insufficient_context is False
    assert 1 <= len(answer.contexts) < 3
    assert len(generator.calls) == 1


def test_relative_signal_flags_a_flat_ranking_but_still_answers():
    generator = FakeGenerator()
    rag = _rag(generator=generator, min_gap=0.99)  # nothing can clear this gap
    rag.ingest([_doc("a.txt", "alpha beta"), _doc("b.txt", "alpha gamma")])
    answer = rag.query("alpha", k=2)
    assert answer.insufficient_context is True
    assert answer.contexts != ()
    assert len(generator.calls) == 1  # the model is told to abstain, not skipped


def test_absolute_floor_flags_a_low_top_score_but_still_answers():
    generator = FakeGenerator()
    rag = _rag(generator=generator, min_score=1.01)  # nothing can clear this
    rag.ingest([_doc("a.txt", "alpha beta")])
    answer = rag.query("alpha", k=2)
    assert answer.insufficient_context is True
    assert answer.contexts != ()
    assert len(generator.calls) == 1
    rag.min_score = None  # the same hits, floor off: sufficient again
    assert rag.query("alpha", k=2).insufficient_context is False


def test_retrieve_is_reproducible_from_the_retriever_and_reranker_alone():
    """plan.md §15 #5: nothing inside ``retrieve()`` a caller cannot do by
    hand from the two public objects it composes."""
    rag = _rag()
    rag.ingest([_doc("a.txt", "alpha beta"), _doc("b.txt", "gamma")])

    by_hand = rag.reranker.rerank(
        "alpha", rag.retriever.retrieve("alpha", k=max(rag.retrieve_k, 1)), 1
    )
    assert rag.retrieve("alpha", k=1) == by_hand
    assert rag.retrieve("alpha", k=1)[0].chunk.doc_id == stable_doc_id("a.txt")


# --- reranking: retrieve-k / rerank-to-n wiring (plan.md §9 Phase F) -----------


def test_default_reranker_is_identity_and_retrieve_k_defaults_to_k():
    rag = _rag(k=3)
    assert isinstance(rag.reranker, IdentityReranker)
    assert rag.retrieve_k == 3


def test_retrieve_k_is_independently_overridable():
    rag = _rag(k=2, retrieve_k=10)
    assert (rag.k, rag.retrieve_k) == (2, 10)


def test_reranker_receives_the_oversampled_pool_and_narrows_to_k():
    fake = FakeReranker()
    rag = _rag(k=2, retrieve_k=4, reranker=fake)
    rag.ingest(
        [
            _doc(f"{i}.txt", "alpha beta gamma delta epsilon")
            for i in range(6)  # more candidates than retrieve_k
        ]
    )

    hits = rag.retrieve("alpha", k=2)

    assert len(hits) == 2
    ([query, n_hits, top_n],) = [list(c) for c in fake.calls]
    assert (query, n_hits, top_n) == ("alpha", 4, 2)


def test_reranker_call_narrows_dynamically_when_a_per_call_k_exceeds_retrieve_k():
    fake = FakeReranker()
    rag = _rag(k=2, retrieve_k=2, reranker=fake)
    rag.ingest([_doc(f"{i}.txt", "alpha beta gamma") for i in range(5)])

    rag.retrieve("alpha", k=5)  # per-call k > retrieve_k

    (call,) = fake.calls
    assert call == ("alpha", 5, 5)  # widened to cover the larger request


def test_reranker_reorders_and_rescores_results():
    rag = _rag(reranker=FakeReranker(score=lambda text: len(text)))
    rag.ingest(
        [_doc("short.txt", "a"), _doc("long.txt", "a much longer chunk of text")]
    )

    hits = rag.retrieve("a", k=2)

    assert [h.chunk.doc_id for h in hits] == [
        stable_doc_id("long.txt"),
        stable_doc_id("short.txt"),
    ]
    assert all(h.source == "rerank:fake" for h in hits)


def test_reranker_failure_degrades_to_retriever_order_with_a_warning(caplog):
    rag = _rag(reranker=FakeReranker(error=RerankError("boom")))
    rag.ingest([_doc("a.txt", "alpha beta"), _doc("b.txt", "alpha gamma")])
    dense_order = rag.retriever.retrieve("alpha", k=rag.retrieve_k)

    with caplog.at_level(logging.WARNING):
        hits = rag.retrieve("alpha", k=2)

    assert hits == dense_order[:2]
    assert any("reranker" in r.message.lower() for r in caplog.records)


def test_retrieve_still_validates_k_at_least_one():
    rag = _rag()
    rag.ingest([_doc("a.txt", "alpha")])
    with pytest.raises(RetrievalError):
        rag.retrieve("alpha", k=0)


def test_abstention_is_judged_on_dense_scores_not_the_rerankers():
    """A real reranker's score is generally on its own, uncalibrated scale
    (Jina's and the local cross-encoder's both are) — plan.md §18 F10's
    ``min_score``/``min_gap`` floors are calibrated against the *embedder's*
    scale, so ``query()`` must judge abstention on the dense hits, never on
    whatever a reranker replaced their scores with."""
    generator = FakeGenerator()

    # A reranker that reports every hit as barely-relevant (0.01) must not
    # make a genuinely good dense match abstain.
    rag = _rag(
        generator=generator,
        min_score=0.1,
        reranker=FakeReranker(score=lambda text: 0.01),
    )
    rag.ingest([_doc("a.txt", "alpha beta gamma")])
    answer = rag.query("alpha beta gamma", k=1)
    assert answer.insufficient_context is False
    assert answer.contexts[0].score == pytest.approx(0.01)  # context still reranked

    # A reranker that reports every hit as highly-relevant (0.99) must not
    # rescue a genuinely poor dense match from abstaining.
    rag2 = _rag(
        generator=generator,
        min_score=1.01,  # nothing on the dense (cosine) scale clears this
        reranker=FakeReranker(score=lambda text: 0.99),
    )
    rag2.ingest([_doc("a.txt", "alpha beta gamma")])
    assert rag2.query("alpha beta gamma", k=1).insufficient_context is True


def test_query_is_reproducible_from_public_functions_alone(tmp_path):
    """plan.md §15 #5: nothing happens inside the facade a user cannot do by hand."""
    generator = FakeGenerator(["by hand [1]", "via facade [1]"])
    counter = HeuristicCounter(warn=False)
    rag = _rag(generator=generator, counter=counter, k=2)
    rag.ingest_path(_write_corpus(tmp_path))
    question = "What do dogs do?"

    # --- by hand, using only public pieces ---
    hits = DenseRetriever(rag.embedder, rag.vectors, rag.docs, k=2).retrieve(question)
    empty = ContextBuilder(1, counter=counter).build([])
    overhead = counter.count(PromptBuilder().build(question, empty).as_text())
    budget = context_budget(
        generator.context_window, generator.max_output_tokens, overhead
    )
    context = ContextBuilder(budget, counter=counter).build(hits)
    prompt = PromptBuilder().build(question, context)
    by_hand = generator.generate(prompt)

    # --- via the facade ---
    answer = rag.query(question)

    assert by_hand.text == "by hand [1]" and answer.text == "via facade [1]"
    assert answer.contexts == context.hits
    assert answer.truncated == context.truncated
    assert answer.insufficient_context == insufficient_context(hits)
    # Same blocks, same order, in both prompts (only the nonce differs).
    assert context.text and context.text in generator.calls[0].user
    assert context.text in generator.calls[1].user


def test_persist_then_reopen_reproduces_retrieval(tmp_path):
    path = tmp_path / "nanorag.sqlite"
    first = _rag(docs=SqliteDocumentStore(path))
    first.ingest_path(_write_corpus(tmp_path / "corpus"))
    before = first.retrieve("goldfish bowls", k=3)
    first.close()

    reopened = _rag(docs=SqliteDocumentStore(path))  # vectors rebuilt from SQLite
    assert reopened.retrieve("goldfish bowls", k=3) == before
    assert reopened.query("goldfish bowls").contexts[0].chunk.doc_id == stable_doc_id(
        "notes/fish.md"
    )
    reopened.close()


def test_usage_reconciliation_warns_when_the_margin_is_exceeded(caplog):
    class Overcounting(FakeGenerator):
        def generate(self, prompt):
            gen = super().generate(prompt)
            return Generation(
                text=gen.text,
                usage=Usage("fake", "m", gen.usage.prompt_tokens * 3, 1, 1),
            )

    rag = _rag(generator=Overcounting())
    rag.ingest([_doc("a.txt", "alpha beta")])
    with caplog.at_level(logging.WARNING, logger="nanorag"):
        rag.query("alpha")
    assert "over the 15% margin" in caplog.text

    caplog.clear()
    rag2 = _rag(generator=FakeGenerator())
    rag2.ingest([_doc("a.txt", "alpha beta")])
    with caplog.at_level(logging.WARNING, logger="nanorag"):
        rag2.query("alpha")
    assert "margin" not in caplog.text


# --- default_generator / from_defaults ----------------------------------------


def _env(**values):
    return lambda name: values.get(name)


def test_default_generator_needs_something_and_names_all_three_options():
    with pytest.raises(ConfigError) as info:
        default_generator(Settings(), env=_env(), ollama_reachable=lambda url: False)
    message = str(info.value)
    assert "GROQ_API_KEY" in message
    assert "GEMINI_API_KEY" in message
    assert "Ollama" in message


def test_default_generator_auto_chains_everything_available(caplog):
    with caplog.at_level(logging.INFO, logger="nanorag"):
        gen = default_generator(
            Settings(),
            env=_env(GROQ_API_KEY="g", GEMINI_API_KEY="k"),
            ollama_reachable=lambda url: True,
        )
    assert isinstance(gen, FallbackGenerator)
    assert [g.provider for g in gen.generators] == ["groq", "gemini", "ollama"]
    assert gen.context_window == 4096  # Ollama's is the tightest
    assert "generator: groq -> gemini -> ollama" in caplog.text


def test_default_generator_single_option_is_returned_bare():
    gen = default_generator(
        Settings(), env=_env(GEMINI_API_KEY="k"), ollama_reachable=lambda url: False
    )
    assert not isinstance(gen, FallbackGenerator)
    assert gen.provider == "gemini"


def test_default_generator_named_preset_uses_only_that_one():
    settings = Settings(generator_preset="ollama")
    gen = default_generator(
        settings, env=_env(GROQ_API_KEY="g"), ollama_reachable=lambda url: True
    )
    assert gen.provider == "ollama"
    with pytest.raises(ConfigError):
        default_generator(
            Settings(generator_preset="gemini"),
            env=_env(GROQ_API_KEY="g"),
            ollama_reachable=lambda url: True,
        )


def test_default_generator_applies_settings_timeouts_and_retries():
    settings = Settings(request_timeout_s=7.0, max_retries=9, retry_base_delay_s=2.0)
    gen = default_generator(
        settings, env=_env(GROQ_API_KEY="g"), ollama_reachable=lambda url: False
    )
    assert gen._backoff.max_retries == 9
    assert gen._backoff.base_delay == 2.0
    assert gen._client.timeout.read == 7.0


def test_ollama_probe_is_true_on_200_and_false_on_connection_failure(monkeypatch):
    from nanorag import pipeline

    def ok(url, timeout):
        return httpx.Response(200, json={"data": []})

    monkeypatch.setattr(httpx, "get", ok)
    assert pipeline._ollama_reachable("http://localhost:11434/v1") is True

    def down(url, timeout):
        raise httpx.ConnectError("refused")

    monkeypatch.setattr(httpx, "get", down)
    assert pipeline._ollama_reachable("http://localhost:11434/v1") is False


def test_from_defaults_wires_persist_dir_cache_and_local_embedder(
    tmp_path, monkeypatch
):
    from nanorag.embeddings import local

    class StubLocal(FakeEmbedder):
        def __init__(self, model_id):
            super().__init__(dim=8, model_id=model_id)

    monkeypatch.setattr(local, "FastEmbedEmbedder", StubLocal)
    persist = tmp_path / ".nanorag"
    rag = Rag.from_defaults(
        persist,
        settings=Settings(embedding_model="stub-model", embed_batch_size=2),
        generator=FakeGenerator(),
        counter=HeuristicCounter(warn=False),
    )
    assert (persist / "nanorag.sqlite").exists()
    assert (persist / "embeddings.sqlite").exists()
    assert rag.embedder.model_id == "stub-model"
    assert rag.min_score is None  # not the default embedder: no floor
    rag.ingest([_doc("a.txt", "alpha beta")])
    assert rag.query("alpha").contexts[0].chunk.doc_id == stable_doc_id("a.txt")
    rag.close()


def test_from_defaults_applies_the_measured_floor_for_the_default_embedder(
    tmp_path, monkeypatch
):
    from nanorag.embeddings import local

    class StubLocal(FakeEmbedder):
        def __init__(self, model_id):
            super().__init__(dim=8, model_id=model_id)

    monkeypatch.setattr(local, "FastEmbedEmbedder", StubLocal)
    rag = Rag.from_defaults(tmp_path, settings=Settings(), generator=FakeGenerator())
    assert rag.min_score == DEFAULT_MIN_SCORE
    rag.close()
    # An explicit value wins over the default.
    rag = Rag.from_defaults(
        tmp_path, settings=Settings(), generator=FakeGenerator(), min_score=None
    )
    assert rag.min_score is None
    rag.close()


def test_from_defaults_without_local_extra_names_the_install(tmp_path, monkeypatch):
    from nanorag.embeddings import local

    def missing(model_id):
        raise ConfigError(
            'local embeddings need the [local] extra: run `uv add "nanorag[local]"`'
        )

    monkeypatch.setattr(local, "FastEmbedEmbedder", missing)
    with pytest.raises(ConfigError, match="nanorag\\[local\\]"):
        Rag.from_defaults(tmp_path, generator=FakeGenerator())


def test_from_defaults_picks_the_generator_when_none_is_given(tmp_path, monkeypatch):
    from nanorag import pipeline
    from nanorag.embeddings import local

    class StubLocal(FakeEmbedder):
        def __init__(self, model_id):
            super().__init__(dim=8, model_id=model_id)

    monkeypatch.setattr(local, "FastEmbedEmbedder", StubLocal)
    monkeypatch.setattr(pipeline, "_ollama_reachable", lambda url, timeout=1.0: False)
    monkeypatch.delenv("GROQ_API_KEY", raising=False)
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    monkeypatch.delenv("NANORAG_PROFILE", raising=False)
    with pytest.raises(ConfigError, match="no generator available"):
        Rag.from_defaults(tmp_path, settings=Settings())
