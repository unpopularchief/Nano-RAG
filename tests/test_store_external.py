"""Unit tests for the external ``VectorStore`` adapters (Phase G, session G2).

No live Qdrant/Postgres service is needed here — only the pure logic that
does not depend on a real connection: missing-extra ``ConfigError``,
constructor argument validation (checked *before* the lazy import, so it
runs the same whether or not the extra happens to be installed), point-id
derivation and DSN/vector-literal parsing. The real, live behaviour
(``-m integration``, needs Docker) is ``tests/test_store_conformance.py``.
"""

from __future__ import annotations

import builtins
import subprocess
import sys

import numpy as np
import pytest

from nanorag.errors import ConfigError, StoreError
from nanorag.store.external.pgvector import (
    PgVectorStore,
    _parse_dsn,
    _parse_vector,
    _vector_literal,
)
from nanorag.store.external.qdrant import QdrantVectorStore, _point_id


def _hide_import(monkeypatch: pytest.MonkeyPatch, hidden_name: str) -> None:
    real_import = builtins.__import__

    def without_it(name, *args, **kwargs):  # type: ignore[no-untyped-def]
        if name == hidden_name or name.startswith(hidden_name + "."):
            raise ImportError(f"{hidden_name} deliberately hidden")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", without_it)


# -- QdrantVectorStore ---------------------------------------------------


def test_missing_qdrant_extra_names_the_install(monkeypatch):
    _hide_import(monkeypatch, "qdrant_client")
    with pytest.raises(ConfigError, match=r"nanorag\[qdrant\].*qdrant-client"):
        QdrantVectorStore(dim=8)


def test_qdrant_store_rejects_bad_dim_before_any_import(monkeypatch):
    _hide_import(monkeypatch, "qdrant_client")
    with pytest.raises(StoreError, match="dim must be >= 1"):
        QdrantVectorStore(dim=0)


def test_qdrant_point_id_is_deterministic_and_distinct():
    first = _point_id("abc123")
    again = _point_id("abc123")
    other = _point_id("def456")
    assert first == again
    assert first != other


# -- PgVectorStore --------------------------------------------------------


def test_missing_pgvector_extra_names_the_install(monkeypatch):
    _hide_import(monkeypatch, "pg8000")
    with pytest.raises(ConfigError, match=r"nanorag\[pgvector\].*pg8000"):
        PgVectorStore(dim=8, dsn="postgresql://user@localhost/db")


def test_pgvector_store_rejects_bad_dim_before_any_import(monkeypatch):
    _hide_import(monkeypatch, "pg8000")
    with pytest.raises(StoreError, match="dim must be >= 1"):
        PgVectorStore(dim=0, dsn="postgresql://user@localhost/db")


def test_pgvector_store_rejects_bad_table_name_before_any_import(monkeypatch):
    _hide_import(monkeypatch, "pg8000")
    with pytest.raises(StoreError, match="bare SQL identifier"):
        PgVectorStore(
            dim=8, dsn="postgresql://user@localhost/db", table_name="bad; drop"
        )


def test_parse_dsn_extracts_connection_kwargs():
    kwargs = _parse_dsn("postgresql://alice:s3cret@dbhost:5544/mydb")
    assert kwargs == {
        "user": "alice",
        "password": "s3cret",
        "host": "dbhost",
        "port": 5544,
        "database": "mydb",
    }


def test_parse_dsn_defaults_host_port_and_database():
    kwargs = _parse_dsn("postgresql://alice@localhost")
    assert kwargs["host"] == "localhost"
    assert kwargs["port"] == 5432
    assert kwargs["database"] is None
    assert kwargs["password"] is None


def test_parse_dsn_rejects_wrong_scheme():
    with pytest.raises(StoreError, match="postgresql://"):
        _parse_dsn("mysql://alice@localhost/db")


def test_parse_dsn_requires_a_user():
    with pytest.raises(StoreError, match="must include a user"):
        _parse_dsn("postgresql://localhost/db")


def test_vector_literal_round_trips_through_parse_vector():
    vector = np.array([0.5, -1.25, 3.0], dtype=np.float32)
    literal = _vector_literal(vector)
    assert literal == "[0.5,-1.25,3.0]"
    np.testing.assert_array_equal(_parse_vector(literal), vector)


def test_importing_store_external_does_not_eagerly_import_the_clients():
    # Fresh process: this pytest process may already have qdrant_client/pg8000
    # imported elsewhere (e.g. by uv-managed extras being present locally).
    code = (
        "import sys; import nanorag.store.external; "
        "assert 'qdrant_client' not in sys.modules; "
        "assert 'pg8000' not in sys.modules"
    )
    subprocess.run([sys.executable, "-c", code], check=True)
