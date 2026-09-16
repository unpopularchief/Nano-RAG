"""``nanorag sync ROOT`` — reconcile the index against a directory tree.

Dry run by default: walks ``ROOT`` and reports what would be added, updated
and deleted without writing anything. ``--apply`` executes the plan.
Offline like ``ingest`` — a :class:`~nanorag.generation.null.NullGenerator`
fills the generator slot, so no key is read and no server is probed.
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import Any

from nanorag.config import Settings
from nanorag.generation.null import NullGenerator
from nanorag.pipeline import DEFAULT_MAX_DELETE_FRACTION, Rag

log = logging.getLogger("nanorag.cli")


def add_parser(
    subparsers: argparse._SubParsersAction[argparse.ArgumentParser],
    name: str,
    *,
    parents: list[argparse.ArgumentParser],
) -> None:
    """Register the ``sync`` subcommand."""
    sub = subparsers.add_parser(
        name,
        parents=parents,
        help="reconcile the index against ROOT: add, update, delete",
        description="Compare the index at --persist-dir against a fresh walk "
        "of ROOT and report additions, updates and deletions. Dry run by "
        "default; --apply executes the plan. Offline, like ingest.",
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
    sub.add_argument(
        "--apply",
        action="store_true",
        help="execute the plan (default: dry run, report only)",
    )
    sub.add_argument(
        "--max-delete-fraction",
        type=float,
        default=DEFAULT_MAX_DELETE_FRACTION,
        metavar="FRACTION",
        help=f"refuse --apply if more than this fraction of the corpus would "
        f"be deleted (default: {DEFAULT_MAX_DELETE_FRACTION})",
    )
    sub.set_defaults(parser=sub)


def run(args: argparse.Namespace, settings: Settings) -> dict[str, Any]:
    """Sync ``args.root``; return the report payload (schema: ``docs/cli.md``)."""
    root = Path(args.root)
    if not root.is_dir():
        args.parser.error(f"not a directory: {root}")
    log.info(
        "syncing %s (glob %s) against %s (apply=%s)",
        root,
        args.glob,
        settings.persist_dir,
        args.apply,
    )
    rag = Rag.from_defaults(settings=settings, generator=NullGenerator())
    try:
        report = rag.sync_path(
            root,
            glob=args.glob,
            ignore=args.ignore,
            apply=args.apply,
            max_delete_fraction=args.max_delete_fraction,
        )
    finally:
        rag.close()
    if report.over_delete_guard:
        # apply=True with the guard tripped raises out of sync_path() before
        # this point is ever reached — so this only ever fires on a dry run,
        # warning before the caller ever passes --apply.
        log.warning(
            "would delete %d document(s), over max_delete_fraction=%s of the "
            "corpus — not applied (dry run); pass --apply to run it anyway, "
            "or a higher --max-delete-fraction",
            len(report.deleted),
            args.max_delete_fraction,
        )
    payload = report.to_dict()
    payload["persist_dir"] = settings.persist_dir
    payload["root"] = str(root)
    payload["glob"] = args.glob
    payload["ignore"] = list(args.ignore)
    return payload


def render(payload: dict[str, Any]) -> str:
    """Render a sync payload as text."""
    mode = "applied" if payload["applied"] else "dry run"
    lines = [
        f"sync {payload['root']} -> {payload['persist_dir']} ({mode}): "
        f"{len(payload['added'])} added, {len(payload['updated'])} updated, "
        f"{len(payload['deleted'])} deleted, {len(payload['unchanged'])} unchanged"
    ]
    for uri in payload["added"]:
        lines.append(f"  + {uri}")
    for uri in payload["updated"]:
        lines.append(f"  ~ {uri}")
    for uri in payload["deleted"]:
        lines.append(f"  - {uri}")
    if payload["over_delete_guard"]:
        # apply=True with the guard tripped raises before a payload is ever
        # built, so a delivered payload only ever reaches here on a dry run.
        lines.append("  ! deletions exceed max_delete_fraction; --apply would refuse")
    return "\n".join(lines) + "\n"


def exit_code(payload: dict[str, Any]) -> int:
    """Return ``0``: the command either completed or raised."""
    return 0
