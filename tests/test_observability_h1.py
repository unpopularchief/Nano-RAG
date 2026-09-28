import io
import json
import logging

import pytest

from nanorag import Rag
from nanorag.errors import ConfigError, ProviderError
from nanorag.generation import FallbackGenerator, Generation
from nanorag.hashing import content_hash, stable_doc_id
from nanorag.observability import JsonFormatter
from nanorag.store import SqliteDocumentStore
from nanorag.tokens import HeuristicCounter
from nanorag.types import Document, Usage
from tests.fakes import FakeEmbedder, FakeGenerator


def _document(uri, text):
    return Document(stable_doc_id(uri), uri, text, content_hash(text))


def _rag(docs, **kwargs):
    return Rag(
        embedder=FakeEmbedder(dim=64),
        generator=FakeGenerator(),
        docs=docs,
        counter=HeuristicCounter(warn=False),
        **kwargs,
    )


def test_query_events_and_explicit_token_prices():
    events = []
    rag = _rag(
        SqliteDocumentStore(":memory:"),
        event_hook=events.append,
        token_prices={("fake", "fake-generator"): (2.0, 4.0)},
    )
    rag.ingest([_document("a.txt", "Cats purr and sleep.")])
    answer = rag.query("What do cats do?")
    usage = answer.usage
    assert usage.prompt_tokens > 0 and usage.completion_tokens > 0
    assert usage.total_tokens == usage.prompt_tokens + usage.completion_tokens
    assert usage.cost_usd == pytest.approx(
        (usage.prompt_tokens * 2 + usage.completion_tokens * 4) / 1_000_000
    )
    assert [event.name for event in events] == ["ingest_complete", "query_complete"]
    assert events[-1].values["cost_usd"] == usage.cost_usd
    rag.close()


def test_missing_provider_counts_use_counter_estimates():
    class NoUsageGenerator(FakeGenerator):
        def generate(self, prompt):
            return Generation(
                "An answer. [1]", Usage("fake", "fake-generator", 0, 0, 0)
            )

    rag = Rag(
        embedder=FakeEmbedder(dim=64),
        generator=NoUsageGenerator(),
        docs=SqliteDocumentStore(":memory:"),
        counter=HeuristicCounter(warn=False),
        token_prices={("fake", "fake-generator"): (1.0, 1.0)},
    )
    rag.ingest([_document("a.txt", "Cats purr and sleep.")])
    usage = rag.query("What do cats do?").usage
    assert usage.prompt_tokens > 0
    assert usage.completion_tokens > 0
    assert usage.total_tokens == usage.prompt_tokens + usage.completion_tokens
    assert usage.cost_usd == pytest.approx(usage.total_tokens / 1_000_000)
    rag.close()


def test_negative_token_price_is_rejected():
    docs = SqliteDocumentStore(":memory:")
    with pytest.raises(ConfigError):
        _rag(
            docs,
            token_prices={("fake", "fake-generator"): (-1.0, 0.0)},
        )
    docs.close()


def test_json_logs_redact_keys_and_never_log_prompt_or_provider_error(monkeypatch):
    secret = "h1-test-secret-value"
    monkeypatch.setenv("GROQ_API_KEY", secret)
    stream = io.StringIO()
    handler = logging.StreamHandler(stream)
    handler.setFormatter(JsonFormatter())
    logger = logging.getLogger("nanorag")
    logger.addHandler(handler)
    old_level = logger.level
    logger.setLevel(logging.INFO)
    try:
        rag = _rag(SqliteDocumentStore(":memory:"))
        rag.ingest([_document("a.txt", f"A private document. {secret}")])
        rag.query(f"What is private? {secret}")
        rag.generator = FallbackGenerator(
            FakeGenerator(responses=[ProviderError(f"provider echoed {secret}")]),
            FakeGenerator(provider="backup"),
        )
        rag.query("Try fallback")
        logger.warning("Authorization: Bearer %s", secret)
        lines = stream.getvalue().splitlines()
        assert lines
        assert all(isinstance(json.loads(line), dict) for line in lines)
        assert secret not in stream.getvalue()
        assert "private document" not in stream.getvalue()
        assert any(json.loads(line).get("event") == "query_complete" for line in lines)
        rag.close()
    finally:
        logger.removeHandler(handler)
        logger.setLevel(old_level)


def test_snapshot_loads_without_sqlite_vector_scan_and_rejects_stale(
    tmp_path, monkeypatch
):
    db = tmp_path / "index.sqlite"
    snapshot = tmp_path / "matrix.npy"
    first = _rag(SqliteDocumentStore(db), snapshot_path=snapshot)
    first.ingest([_document("a.txt", "Cats purr and sleep.")])
    first.save_snapshot()
    expected = len(first.vectors)
    expected_hits = first.retrieve("What do cats do?")
    first.close()

    reopened_docs = SqliteDocumentStore(db)

    def unexpected_scan():
        raise AssertionError("snapshot should skip SQLite vector scan")
        yield

    monkeypatch.setattr(reopened_docs, "iter_embeddings", unexpected_scan)
    reopened = _rag(reopened_docs, snapshot_path=snapshot)
    assert len(reopened.vectors) == expected
    assert reopened.retrieve("What do cats do?") == expected_hits
    reopened.close()

    updated = _rag(SqliteDocumentStore(db), snapshot_path=snapshot)
    updated.ingest([_document("b.txt", "Dogs bark and run.")])
    updated.close()
    stale = _rag(SqliteDocumentStore(db), snapshot_path=snapshot)
    assert len(stale.vectors) > expected
    stale.close()


def test_snapshot_corruption_rebuilds_from_sqlite(tmp_path):
    db = tmp_path / "index.sqlite"
    snapshot = tmp_path / "matrix.npy"
    first = _rag(SqliteDocumentStore(db), snapshot_path=snapshot)
    first.ingest([_document("a.txt", "Cats purr and sleep.")])
    first.save_snapshot()
    expected = len(first.vectors)
    first.close()
    snapshot.write_bytes(b"corrupt")
    reopened = _rag(SqliteDocumentStore(db), snapshot_path=snapshot)
    assert len(reopened.vectors) == expected
    reopened.close()


def test_file_backed_sqlite_uses_wal(tmp_path):
    docs = SqliteDocumentStore(tmp_path / "index.sqlite")
    assert docs._conn.execute("PRAGMA journal_mode").fetchone()[0] == "wal"
    docs.close()


def test_embedding_revision_advances_after_delete(tmp_path):
    rag = _rag(SqliteDocumentStore(tmp_path / "index.sqlite"))
    rag.ingest([_document("a.txt", "Cats purr and sleep.")])
    before = rag.docs.embedding_revision()
    rag.delete_document(stable_doc_id("a.txt"))
    assert rag.docs.embedding_revision() > before
    rag.close()
