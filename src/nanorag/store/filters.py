"""The metadata filter grammar and its compilation to bound-parameter SQL.

A filter is a small, flat, Mongo-shaped mapping that a retriever evaluates
**before** similarity search, so the top-*k* is taken over the chunks that
match rather than filtered afterwards (plan.md §6 "Retriever": metadata
*pre*-filters; §15 #7). It compiles to a SQL ``WHERE`` fragment over the
``chunks`` × ``documents`` join, with every operand bound as a parameter —
never string-interpolated (plan.md §13).

Grammar
-------
::

    filter    := { entry, ... }               # every entry must hold (AND)
    entry     := key : scalar                 # shorthand for { "$eq": scalar }
               | key : { op : operand, ... }  # every op must hold (AND)
               | "$and" : [ filter, ... ]
               | "$or"  : [ filter, ... ]
               | "$not" : filter
    op        := "$eq" | "$ne" | "$gt" | "$gte" | "$lt" | "$lte"
               | "$in" | "$nin" | "$prefix" | "$exists"

Keys
----
Four keys address columns directly: ``chunk_id``, ``doc_id``, ``ordinal``
(the chunk's) and ``source_uri`` (the owning document's). Every other key is
a metadata key, looked up in the chunk's metadata first and then the
document's — so a document-level ``{"lang": "en"}`` applies to all of its
chunks unless a chunk overrides it.

Semantics
---------
- A key that is absent (or ``None``, which JSON cannot tell from absent)
  satisfies **no** comparison operator — not ``$ne``, not ``$nin``. Use
  ``{"$exists": False}`` to match absence, or ``$not`` to negate a whole
  sub-filter.
- ``$gt`` / ``$gte`` / ``$lt`` / ``$lte`` compare numbers with numbers and
  text with text; a value of the other kind never matches. Booleans are
  stored as ``0`` / ``1`` and compare as integers (``True == 1``), exactly
  as in Python.
- ``$prefix`` matches text values starting with the operand.
- ``$in`` / ``$nin`` take a list of scalars; ``$in: []`` matches nothing.

Cost: metadata keys are ``json_extract`` scans over the chunk table, not
index lookups (plan.md §18 F8) — a filtered query is bounded by the corpus
size, not by *k*.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from typing import Any

from nanorag.errors import RetrievalError
from nanorag.types import JsonScalar

#: A filter, as accepted by ``compile_filter`` and the retrievers.
Filter = Mapping[str, Any]

#: Keys that address a column rather than a metadata entry, with the SQL
#: expression and the Python type an operand must have to compare cleanly
#: (SQLite applies column affinity when comparing, so ``ordinal = '3'`` would
#: silently coerce — refusing the operand at compile time is safer).
_COLUMNS: dict[str, tuple[str, type]] = {
    "chunk_id": ("chunks.chunk_id", str),
    "doc_id": ("chunks.doc_id", str),
    "ordinal": ("chunks.ordinal", int),
    "source_uri": ("documents.source_uri", str),
}

_COMBINATORS = frozenset({"$and", "$or", "$not"})
_SCALAR_OPS = frozenset({"$eq", "$ne", "$gt", "$gte", "$lt", "$lte", "$prefix"})
_LIST_OPS = frozenset({"$in", "$nin"})
_ALL_OPS = _SCALAR_OPS | _LIST_OPS | {"$exists"}
_ORDER_OPS = {"$gt": ">", "$gte": ">=", "$lt": "<", "$lte": "<="}


def compile_filter(filter: Filter) -> tuple[str, list[JsonScalar]]:
    """Compile *filter* to a SQL predicate over ``chunks`` joined to ``documents``.

    Parameters
    ----------
    filter
        A mapping in the grammar described in the module docstring.

    Returns
    -------
    tuple[str, list[JsonScalar]]
        A parenthesised SQL boolean expression referencing the ``chunks``
        and ``documents`` tables, and the parameters to bind to its ``?``
        placeholders, in order. Every leaf evaluates to exactly ``0`` or
        ``1`` (never ``NULL``), so ``$not`` is a plain negation.

    Raises
    ------
    RetrievalError
        *filter* is not in the grammar: an unknown operator, an operand of
        the wrong type, a non-string or reserved key, or a malformed
        combinator.

    """
    if not isinstance(filter, Mapping):
        raise RetrievalError("filter must be a mapping", got=type(filter).__name__)
    return _compile_mapping(filter)


def _compile_mapping(filter: Filter) -> tuple[str, list[JsonScalar]]:
    """Compile one ``{ entry, ... }`` level; entries are ANDed."""
    clauses: list[str] = []
    params: list[JsonScalar] = []
    for key, value in filter.items():
        if not isinstance(key, str) or not key:
            raise RetrievalError("filter keys must be non-empty strings", key=key)
        if key in _COMBINATORS:
            sql, sub_params = _compile_combinator(key, value)
        elif key.startswith("$"):
            raise RetrievalError(
                "unknown combinator (keys starting with '$' are reserved)",
                key=key,
                allowed=sorted(_COMBINATORS),
            )
        else:
            sql, sub_params = _compile_key(key, value)
        clauses.append(sql)
        params.extend(sub_params)
    if not clauses:
        return "(1)", []
    return "(" + " AND ".join(clauses) + ")", params


def _compile_combinator(key: str, value: object) -> tuple[str, list[JsonScalar]]:
    """Compile ``$and`` / ``$or`` (lists of filters) or ``$not`` (one filter)."""
    if key == "$not":
        if not isinstance(value, Mapping):
            raise RetrievalError("$not takes a single filter mapping", key=key)
        sql, inner_params = _compile_mapping(value)
        return f"(NOT {sql})", inner_params

    if not isinstance(value, Sequence) or isinstance(value, str | bytes):
        raise RetrievalError(f"{key} takes a list of filter mappings", key=key)
    if not value:
        raise RetrievalError(f"{key} needs at least one filter", key=key)
    parts: list[str] = []
    params: list[JsonScalar] = []
    for item in value:
        if not isinstance(item, Mapping):
            raise RetrievalError(f"{key} items must be filter mappings", key=key)
        sql, sub_params = _compile_mapping(item)
        parts.append(sql)
        params.extend(sub_params)
    joiner = " AND " if key == "$and" else " OR "
    return "(" + joiner.join(parts) + ")", params


def _compile_key(key: str, value: object) -> tuple[str, list[JsonScalar]]:
    """Compile ``key: scalar`` or ``key: {op: operand, ...}``."""
    expr, expr_params, column_type = _value_expr(key)
    if not isinstance(value, Mapping):
        ops: Mapping[str, object] = {"$eq": value}
    else:
        ops = value
    if not ops:
        raise RetrievalError("operator mapping must not be empty", key=key)

    clauses: list[str] = []
    params: list[JsonScalar] = []
    for op, operand in ops.items():
        if not isinstance(op, str) or op not in _ALL_OPS:
            raise RetrievalError(
                "unknown filter operator", key=key, op=op, allowed=sorted(_ALL_OPS)
            )
        sql, op_params = _compile_op(key, op, operand, expr, expr_params, column_type)
        clauses.append(sql)
        params.extend(op_params)
    return "(" + " AND ".join(clauses) + ")", params


def _value_expr(key: str) -> tuple[str, list[JsonScalar], type | None]:
    """Return the SQL expression, its params, and the column type for *key*.

    A metadata key compiles to ``COALESCE(json_extract(chunk), json_extract(
    document))`` — chunk metadata wins, document metadata is the fallback —
    with the JSON path bound as a parameter. The column type is ``None`` for
    metadata (any JSON scalar may be stored there).
    """
    if key in _COLUMNS:
        column, column_type = _COLUMNS[key]
        return column, [], column_type
    if '"' in key:
        raise RetrievalError(
            "metadata filter keys must not contain a double quote", key=key
        )
    path = f'$."{key}"'
    return (
        "COALESCE(json_extract(chunks.metadata, ?), "
        "json_extract(documents.metadata, ?))",
        [path, path],
        None,
    )


def _compile_op(
    key: str,
    op: str,
    operand: object,
    expr: str,
    expr_params: list[JsonScalar],
    column_type: type | None,
) -> tuple[str, list[JsonScalar]]:
    """Compile one ``op: operand`` leaf into a ``0``/``1`` SQL expression."""
    if op == "$exists":
        if not isinstance(operand, bool):
            raise RetrievalError("$exists takes a bool", key=key, got=operand)
        test = "IS NOT NULL" if operand else "IS NULL"
        return f"({expr} {test})", list(expr_params)

    if op in _LIST_OPS:
        if not isinstance(operand, Sequence) or isinstance(operand, str | bytes):
            raise RetrievalError(f"{op} takes a list of scalars", key=key, op=op)
        values = [_check_operand(key, op, item, column_type) for item in operand]
        if not values:
            # Nothing is in an empty list; every present value is outside it.
            if op == "$in":
                return "(0)", []
            return f"({expr} IS NOT NULL)", list(expr_params)
        placeholders = ", ".join("?" for _ in values)
        negate = "NOT " if op == "$nin" else ""
        return (
            f"(COALESCE({expr} {negate}IN ({placeholders}), 0))",
            [*expr_params, *values],
        )

    scalar = _check_operand(key, op, operand, column_type)
    if op == "$eq":
        return f"(COALESCE({expr} = ?, 0))", [*expr_params, scalar]
    if op == "$ne":
        return f"(COALESCE({expr} <> ?, 0))", [*expr_params, scalar]
    if op == "$prefix":
        if not isinstance(scalar, str):
            raise RetrievalError("$prefix takes a string", key=key, got=scalar)
        # substr counts characters, like len(); no LIKE-escaping needed.
        return (
            f"(COALESCE(typeof({expr}) = 'text' AND substr({expr}, 1, ?) = ?, 0))",
            [*expr_params, *expr_params, len(scalar), scalar],
        )

    # $gt / $gte / $lt / $lte: compare like with like only. SQLite orders
    # every integer/real below every text, so without the typeof guard a
    # `$gt: 5` would match every string value.
    if isinstance(scalar, bool):
        raise RetrievalError(f"{op} takes a number or a string", key=key, got=scalar)
    kind = "= 'text'" if isinstance(scalar, str) else "IN ('integer', 'real')"
    return (
        f"(COALESCE(typeof({expr}) {kind} AND {expr} {_ORDER_OPS[op]} ?, 0))",
        [*expr_params, *expr_params, scalar],
    )


def _check_operand(
    key: str, op: str, operand: object, column_type: type | None
) -> JsonScalar:
    """Validate a comparison operand and return it typed as a JSON scalar."""
    if operand is None:
        raise RetrievalError(
            f"{op} operand must not be None (use $exists)", key=key, op=op
        )
    if not isinstance(operand, str | int | float):
        raise RetrievalError(
            f"{op} operand must be a JSON scalar",
            key=key,
            op=op,
            got=type(operand).__name__,
        )
    if isinstance(operand, float) and not math.isfinite(operand):
        raise RetrievalError(f"{op} operand must be finite", key=key, op=op)
    if column_type is not None and (
        not isinstance(operand, column_type) or isinstance(operand, bool)
    ):
        raise RetrievalError(
            f"{key} is a {column_type.__name__} column",
            key=key,
            op=op,
            got=type(operand).__name__,
        )
    return operand
