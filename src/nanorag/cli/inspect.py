"""``nanorag inspect`` — what is in the index, without loading a model.

Opens the SQLite store under ``--persist-dir`` directly: the embedding
model and dimension the index was built with, document and chunk counts,
and one row per document. No embedder, no generator, no network — and
nothing is created: a missing index is a ``StoreError`` (exit 5), never
a fresh empty database.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

from nanorag.config import Settings
from nanorag.errors import StoreError
from nanorag.store.sqlite_docs import SqliteDocumentStore

#: The document store's file name under ``persist_dir`` (see ``Rag.from_defaults``).
INDEX_FILE = "nanorag.sqlite"


def add_parser(
    subparsers: argparse._SubParsersAction[argparse.ArgumentParser],
    name: str,
    *,
    parents: list[argparse.ArgumentParser],
) -> None:
    """Register the ``inspect`` subcommand."""
    sub = subparsers.add_parser(
        name,
        parents=parents,
        help="show what the index holds (model, counts, documents)",
        description="Report the index at --persist-dir: embedding model and "
        "dimension, document and chunk counts, one line per document. "
        "Offline; loads no model.",
    )
    sub.set_defaults(parser=sub)


def run(args: argparse.Namespace, settings: Settings) -> dict[str, Any]:
    """Describe the index; return the payload (schema: ``docs/cli.md``)."""
    path = Path(settings.persist_dir) / INDEX_FILE
    if not path.is_file():
        raise StoreError(
            f"no index at {settings.persist_dir} — run `nanorag ingest ROOT` first",
            path=str(path),
        )
    with SqliteDocumentStore(path) as docs:
        meta = docs.index_meta()
        documents = [
            {
                "doc_id": d.doc_id,
                "source_uri": d.source_uri,
                "content_hash": d.content_hash,
                "chunks": docs.count_chunks(d.doc_id),
                "metadata": dict(d.metadata),
            }
            for d in sorted(docs.iter_documents(), key=lambda d: d.source_uri)
        ]
        return {
            "persist_dir": settings.persist_dir,
            "index": {"model_id": meta[0], "dim": meta[1]} if meta else None,
            "documents": docs.count_documents(),
            "chunks": docs.count_chunks(),
            "sources": documents,
        }


def render(payload: dict[str, Any]) -> str:
    """Render an inspect payload as text."""
    index = payload["index"]
    model = f"{index['model_id']} ({index['dim']}-dim)" if index else "none yet"
    lines = [
        f"index: {payload['persist_dir']}",
        f"embedding model: {model}",
        f"documents: {payload['documents']}, chunks: {payload['chunks']}",
    ]
    if payload["sources"]:
        lines += ["", "documents (chunks, id, source):"]
        for d in payload["sources"]:
            lines.append(f"  {d['chunks']:>5}  {d['doc_id']}  {d['source_uri']}")
    return "\n".join(lines) + "\n"
