"""The filter grammar: compile-time validation, per-operator semantics, and a
Hypothesis check that the SQL the compiler emits agrees with a plain-Python
reference evaluator on random metadata and random filters."""

from collections.abc import Mapping

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from nanorag.errors import RetrievalError
from nanorag.hashing import chunk_id, content_hash, stable_doc_id
from nanorag.store import SqliteDocumentStore, compile_filter
from nanorag.types import Chunk, Document


def _doc(source_uri: str, metadata: Mapping | None = None) -> Document:
    text = f"text of {source_uri}"
    return Document(
        doc_id=stable_doc_id(source_uri),
        source_uri=source_uri,
        text=text,
        content_hash=content_hash(text),
        metadata=metadata or {},
    )


def _chunk(doc: Document, ordinal: int, metadata: Mapping | None = None) -> Chunk:
    text = f"{doc.source_uri}#{ordinal}"
    return Chunk(
        chunk_id=chunk_id(doc.doc_id, ordinal, text),
        doc_id=doc.doc_id,
        ordinal=ordinal,
        text=text,
        start_char=0,
        end_char=len(text),
        token_count=1,
        metadata=metadata or {},
    )


@pytest.fixture
def store():
    """A small corpus with document-level and chunk-level metadata."""
    s = SqliteDocumentStore(":memory:")
    api = _doc("docs/api/retriever.md", {"lang": "en", "year": 2021})
    notes = _doc("notes/todo.txt", {"lang": "de", "year": 2019, "draft": True})
    s.upsert_document(
        api,
        [
            _chunk(api, 0, {"nanorag.heading_path": "Intro > Filters"}),
            _chunk(api, 1, {"lang": "fr"}),  # chunk overrides document
        ],
    )
    s.upsert_document(notes, [_chunk(notes, 0), _chunk(notes, 1, {"year": None})])
    yield s
    s.close()


def _texts(store: SqliteDocumentStore, filter) -> list[str]:
    ids = store.filter_chunk_ids(filter)
    return sorted(c.text for c in store.get_chunks_by_ids(list(ids)).values())


# --- operators, one at a time -----------------------------------------------


def test_equality_shorthand_on_a_metadata_key(store):
    assert _texts(store, {"lang": "en"}) == ["docs/api/retriever.md#0"]


def test_chunk_metadata_overrides_document_metadata(store):
    assert _texts(store, {"lang": "fr"}) == ["docs/api/retriever.md#1"]


def test_none_in_chunk_metadata_falls_back_to_the_document_value(store):
    # JSON null and absent are indistinguishable to json_extract, so the
    # chunk's `year: None` falls through to the document's 2019.
    assert _texts(store, {"year": 2019}) == ["notes/todo.txt#0", "notes/todo.txt#1"]


def test_prefix_on_source_uri_column(store):
    assert _texts(store, {"source_uri": {"$prefix": "docs/api/"}}) == [
        "docs/api/retriever.md#0",
        "docs/api/retriever.md#1",
    ]


def test_prefix_on_a_metadata_key_with_a_dot_in_its_name(store):
    assert _texts(store, {"nanorag.heading_path": {"$prefix": "Intro"}}) == [
        "docs/api/retriever.md#0"
    ]


def test_prefix_is_not_a_like_pattern(store):
    # `%` and `_` are literal characters, not wildcards.
    assert _texts(store, {"source_uri": {"$prefix": "docs%"}}) == []
    assert _texts(store, {"source_uri": {"$prefix": "docs_"}}) == []


def test_ordinal_column(store):
    assert _texts(store, {"ordinal": 1}) == [
        "docs/api/retriever.md#1",
        "notes/todo.txt#1",
    ]


def test_doc_id_and_chunk_id_columns(store):
    doc = _doc("notes/todo.txt")
    assert _texts(store, {"doc_id": doc.doc_id}) == [
        "notes/todo.txt#0",
        "notes/todo.txt#1",
    ]
    cid = _chunk(doc, 0).chunk_id
    assert _texts(store, {"chunk_id": cid}) == ["notes/todo.txt#0"]


def test_ne_does_not_match_an_absent_key(store):
    # Neither chunk of notes/ has heading_path; $ne never matches absence.
    assert _texts(store, {"nanorag.heading_path": {"$ne": "x"}}) == [
        "docs/api/retriever.md#0"
    ]


def test_not_does_match_an_absent_key(store):
    assert _texts(store, {"$not": {"nanorag.heading_path": "x"}}) == [
        "docs/api/retriever.md#0",
        "docs/api/retriever.md#1",
        "notes/todo.txt#0",
        "notes/todo.txt#1",
    ]


def test_exists(store):
    assert _texts(store, {"nanorag.heading_path": {"$exists": True}}) == [
        "docs/api/retriever.md#0"
    ]
    assert _texts(store, {"draft": {"$exists": False}}) == [
        "docs/api/retriever.md#0",
        "docs/api/retriever.md#1",
    ]


def test_range_operators_on_numbers(store):
    assert _texts(store, {"year": {"$gte": 2020}}) == [
        "docs/api/retriever.md#0",
        "docs/api/retriever.md#1",
    ]
    assert _texts(store, {"year": {"$gt": 2019, "$lt": 2022}}) == [
        "docs/api/retriever.md#0",
        "docs/api/retriever.md#1",
    ]
    assert _texts(store, {"year": {"$lte": 2019}}) == [
        "notes/todo.txt#0",
        "notes/todo.txt#1",
    ]


def test_range_operators_never_compare_numbers_with_text(store):
    # SQLite orders every number below every string; the typeof guard stops
    # `$gt: "a"` from matching numeric years (and vice versa).
    assert _texts(store, {"year": {"$gt": "a"}}) == []
    assert _texts(store, {"year": {"$lt": "a"}}) == []
    assert _texts(store, {"lang": {"$gt": 0}}) == []


def test_range_operators_on_text(store):
    assert _texts(store, {"lang": {"$gt": "de", "$lte": "en"}}) == [
        "docs/api/retriever.md#0"
    ]


def test_booleans_compare_as_integers(store):
    assert _texts(store, {"draft": True}) == ["notes/todo.txt#0", "notes/todo.txt#1"]
    assert _texts(store, {"draft": 1}) == ["notes/todo.txt#0", "notes/todo.txt#1"]
    assert _texts(store, {"draft": False}) == []


def test_in_and_nin(store):
    assert _texts(store, {"lang": {"$in": ["fr", "de"]}}) == [
        "docs/api/retriever.md#1",
        "notes/todo.txt#0",
        "notes/todo.txt#1",
    ]
    assert _texts(store, {"lang": {"$nin": ["fr", "de"]}}) == [
        "docs/api/retriever.md#0"
    ]


def test_in_empty_list_matches_nothing_and_nin_empty_matches_present(store):
    assert _texts(store, {"lang": {"$in": []}}) == []
    assert _texts(store, {"draft": {"$nin": []}}) == [
        "notes/todo.txt#0",
        "notes/todo.txt#1",
    ]


def test_and_or_combinators(store):
    assert _texts(store, {"$or": [{"lang": "fr"}, {"lang": "de"}]}) == [
        "docs/api/retriever.md#1",
        "notes/todo.txt#0",
        "notes/todo.txt#1",
    ]
    assert _texts(store, {"$and": [{"lang": "de"}, {"ordinal": 0}]}) == [
        "notes/todo.txt#0"
    ]


def test_multiple_keys_are_anded(store):
    assert _texts(store, {"lang": "en", "ordinal": 0}) == ["docs/api/retriever.md#0"]
    assert _texts(store, {"lang": "en", "ordinal": 1}) == []


def test_filter_matching_nothing_returns_empty_set_not_an_error(store):
    assert store.filter_chunk_ids({"no_such_key": "x"}) == set()
    assert store.filter_chunk_ids({"source_uri": "missing.txt"}) == set()


def test_empty_filter_matches_everything(store):
    assert len(store.filter_chunk_ids({})) == 4


def test_get_chunks_by_ids_omits_unknown_ids_and_batches(store):
    all_ids = sorted(store.filter_chunk_ids({}))
    found = store.get_chunks_by_ids([*all_ids, "not-a-chunk"])
    assert sorted(found) == all_ids
    assert store.get_chunks_by_ids([]) == {}


def test_get_chunks_by_ids_handles_more_ids_than_one_in_batch(tmp_path):
    from nanorag.store import sqlite_docs

    s = SqliteDocumentStore(tmp_path / "db.sqlite3")
    doc = _doc("big.txt")
    chunks = [_chunk(doc, i) for i in range(sqlite_docs._IN_BATCH + 7)]
    s.upsert_document(doc, chunks)
    found = s.get_chunks_by_ids([c.chunk_id for c in chunks])
    assert len(found) == len(chunks)
    s.close()


# --- compile-time validation -------------------------------------------------


@pytest.mark.parametrize(
    "bad",
    [
        "not a mapping",
        {"": 1},
        {3: 1},
        {"$unknown": [{"a": 1}]},
        {"a": {"$regex": "x"}},
        {"a": {}},
        {"a": {"$eq": None}},
        {"a": {"$eq": [1, 2]}},
        {"a": {"$eq": float("nan")}},
        {"a": {"$in": "abc"}},
        {"a": {"$in": [None]}},
        {"a": {"$gt": True}},
        {"a": {"$prefix": 5}},
        {"a": {"$exists": 1}},
        {"$and": {"a": 1}},
        {"$and": []},
        {"$and": [1]},
        {"$or": "x"},
        {"$not": [{"a": 1}]},
        {'has"quote': 1},
        {"ordinal": "3"},
        {"ordinal": 1.5},
        {"ordinal": True},
        {"ordinal": {"$prefix": "1"}},
        {"source_uri": 5},
        {"doc_id": {"$in": ["ok", 7]}},
    ],
)
def test_malformed_filters_raise_retrieval_error(bad):
    with pytest.raises(RetrievalError):
        compile_filter(bad)


def test_compiled_sql_binds_every_operand_as_a_parameter():
    sql, params = compile_filter({"source_uri": {"$prefix": "x'; DROP TABLE"}})
    assert "DROP" not in sql
    assert "x'; DROP TABLE" in params
    assert sql.count("?") == len(params)


def test_column_names_are_never_taken_from_the_filter():
    sql, params = compile_filter({"documents.text": "x"})
    assert "documents.text" not in sql
    assert '$."documents.text"' in params


# --- SQL agrees with a Python reference evaluator ---------------------------


def _lookup(key, chunk: Chunk, doc: Document):
    if key == "chunk_id":
        return chunk.chunk_id
    if key == "doc_id":
        return chunk.doc_id
    if key == "ordinal":
        return chunk.ordinal
    if key == "source_uri":
        return doc.source_uri
    value = chunk.metadata.get(key)
    return doc.metadata.get(key) if value is None else value


def _same_kind(a, b) -> bool:
    return isinstance(a, str) == isinstance(b, str)


def _op_matches(op, operand, value) -> bool:
    if op == "$exists":
        return (value is not None) is operand
    if value is None:
        return False
    if op == "$eq":
        return value == operand
    if op == "$ne":
        return value != operand
    if op == "$in":
        return value in operand
    if op == "$nin":
        return value not in operand
    if op == "$prefix":
        return isinstance(value, str) and value.startswith(operand)
    if not _same_kind(value, operand):
        return False
    return {
        "$gt": value > operand,
        "$gte": value >= operand,
        "$lt": value < operand,
        "$lte": value <= operand,
    }[op]


def reference_matches(filter, chunk: Chunk, doc: Document) -> bool:
    """Evaluate *filter* in plain Python — the oracle for the SQL compiler."""
    for key, value in filter.items():
        if key == "$and":
            if not all(reference_matches(f, chunk, doc) for f in value):
                return False
        elif key == "$or":
            if not any(reference_matches(f, chunk, doc) for f in value):
                return False
        elif key == "$not":
            if reference_matches(value, chunk, doc):
                return False
        else:
            ops = value if isinstance(value, Mapping) else {"$eq": value}
            stored = _lookup(key, chunk, doc)
            if not all(_op_matches(op, operand, stored) for op, operand in ops.items()):
                return False
    return True


# Small pools, so stored values and filter operands collide often — an
# oracle only bites if equality, ties and boundaries actually occur.
_KEYS = ["a", "b", "c.d"]
_TEXT = st.sampled_from(["", "a", "ab", "b", "x/y", chr(0x00E9), chr(0x4E2D)])
_NUMBER = st.sampled_from([-1, 0, 1, 2, 0.5, 1.0, -1.5])
_SCALAR = st.one_of(_TEXT, _NUMBER, st.booleans(), st.none())
_METADATA = st.dictionaries(st.sampled_from(_KEYS), _SCALAR, max_size=3)

_OPERAND = st.one_of(_TEXT, _NUMBER, st.booleans())
_ORDER_OPERAND = st.one_of(_TEXT, _NUMBER)


def _leaf() -> st.SearchStrategy:
    key_ops = [
        _OPERAND,  # equality shorthand
        st.fixed_dictionaries({"$eq": _OPERAND}),
        st.fixed_dictionaries({"$ne": _OPERAND}),
        st.fixed_dictionaries({"$gt": _ORDER_OPERAND}),
        st.fixed_dictionaries({"$gte": _ORDER_OPERAND}),
        st.fixed_dictionaries({"$lt": _ORDER_OPERAND}),
        st.fixed_dictionaries({"$lte": _ORDER_OPERAND}),
        st.fixed_dictionaries({"$in": st.lists(_OPERAND, max_size=3)}),
        st.fixed_dictionaries({"$nin": st.lists(_OPERAND, max_size=3)}),
        st.fixed_dictionaries({"$prefix": _TEXT}),
        st.fixed_dictionaries({"$exists": st.booleans()}),
        st.fixed_dictionaries({"$gte": _NUMBER, "$lt": _NUMBER}),
    ]
    metadata_leaves = [st.tuples(st.sampled_from(_KEYS), ops) for ops in key_ops]
    column_leaves = [
        st.tuples(st.just("ordinal"), st.integers(min_value=0, max_value=3)),
        st.tuples(
            st.just("source_uri"),
            st.fixed_dictionaries({"$prefix": st.sampled_from(["d", "doc", "n"])}),
        ),
    ]
    # Flat, equal-weight choice over every leaf shape, so each operator is
    # exercised often enough for the oracle to bite.
    return st.one_of(*metadata_leaves, *column_leaves).map(lambda kv: {kv[0]: kv[1]})


_FILTER = st.recursive(
    _leaf(),
    lambda inner: st.one_of(
        st.lists(inner, min_size=1, max_size=3).map(lambda fs: {"$and": fs}),
        st.lists(inner, min_size=1, max_size=3).map(lambda fs: {"$or": fs}),
        inner.map(lambda f: {"$not": f}),
    ),
    max_leaves=4,
)


@settings(max_examples=300, deadline=None)
@given(
    doc_metas=st.lists(_METADATA, min_size=1, max_size=3),
    chunk_metas=st.lists(_METADATA, min_size=1, max_size=3),
    filter=_FILTER,
)
def test_compiled_sql_agrees_with_the_python_reference(doc_metas, chunk_metas, filter):
    store = SqliteDocumentStore(":memory:")
    try:
        expected: set[str] = set()
        for i, doc_meta in enumerate(doc_metas):
            doc = _doc(f"{'docs' if i % 2 else 'notes'}/{i}.txt", doc_meta)
            chunks = [_chunk(doc, j, meta) for j, meta in enumerate(chunk_metas)]
            store.upsert_document(doc, chunks)
            expected |= {
                c.chunk_id for c in chunks if reference_matches(filter, c, doc)
            }
        assert store.filter_chunk_ids(filter) == expected
    finally:
        store.close()
