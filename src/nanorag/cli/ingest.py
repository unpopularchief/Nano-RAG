"""``nanorag ingest ROOT`` — load, chunk, embed and persist a directory.

Fully offline: local embeddings, SQLite under ``--persist-dir``, and a
:class:`~nanorag.generation.NullGenerator` in the generator slot, so no
key is read and no server probed for work that needs neither. The walk
is the library's own ``DirectoryLoader``; one file failing never stops
it, and the failures are part of the report rather than an exit code.
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import Any

from nanorag.config import Settings
from nanorag.generation.null import NullGenerator
from nanorag.pipeline import Rag

log = logging.getLogger("nanorag.cli")


def add_parser(
    subparsers: argparse._SubParsersAction[argparse.ArgumentParser],
    name: str,
    *,
    parents: list[argparse.ArgumentParser],
) -> None:
    """Register the ``ingest`` subcommand."""
    sub = subparsers.add_parser(
        name,
        parents=parents,
        help="load, chunk, embed and persist every matching file under ROOT",
        description="Ingest a directory into the index at --persist-dir. "
        "Offline: local embeddings, no generator, no key.",
    )
    sub.add_argument("root", help="directory to walk")
    sub.add_argument(
        "--glob",
        default="**/*",
        help="gitignore-style pattern files must match, relative to ROOT "
        "(default: '**/*')",
    )
    sub.add_argument(
        "--ignore",
        action="append",
        default=[],
        metavar="PATTERN",
        help="pattern to exclude entirely; repeatable",
    )
    sub.set_defaults(parser=sub)


def run(args: argparse.Namespace, settings: Settings) -> dict[str, Any]:
    """Ingest ``args.root``; return the report payload (schema: ``docs/cli.md``)."""
    root = Path(args.root)
    if not root.is_dir():
        args.parser.error(f"not a directory: {root}")
    log.info("ingesting %s (glob %s) into %s", root, args.glob, settings.persist_dir)
    rag = Rag.from_defaults(settings=settings, generator=NullGenerator())
    try:
        report = rag.loader.load_path(root, glob=args.glob, ignore=args.ignore)
        chunks = rag.ingest(report.loaded)
    finally:
        rag.close()
    return {
        "persist_dir": settings.persist_dir,
        "root": str(root),
        "glob": args.glob,
        "ignore": list(args.ignore),
        "chunks": chunks,
        "loaded": [
            {"doc_id": d.doc_id, "source_uri": d.source_uri} for d in report.loaded
        ],
        "skipped": [i.to_dict() for i in report.skipped],
        "failed": [i.to_dict() for i in report.failed],
    }


def render(payload: dict[str, Any]) -> str:
    """Render an ingest payload as text."""
    lines = [
        f"ingested {len(payload['loaded'])} documents ({payload['chunks']} chunks) "
        f"from {payload['root']} into {payload['persist_dir']}; "
        f"skipped {len(payload['skipped'])}, failed {len(payload['failed'])}"
    ]
    for issue in payload["failed"]:
        lines.append(f"  failed  {issue['source_uri']}: {issue['reason']}")
    return "\n".join(lines) + "\n"
