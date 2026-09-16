"""The ``nanorag`` command: argument parsing, dispatch, exit codes.

Five subcommands — ``ingest``, ``query``, ``inspect``, ``eval``, ``sync`` —
each a module in this package with the same three-function shape:
``run(args, settings)`` returns a JSON-serialisable payload,
``render(payload)`` turns that payload into the human-readable text, and
``exit_code(payload)`` says how the process ends once it has been printed.
``--json`` prints the payload itself, so **the text output is a view of
the JSON, never a superset of it** — the schema documented in
``docs/cli.md`` is the whole contract.

Two stream rules keep ``--json`` machine-readable:

- stdout carries the result and nothing else;
- every log line, warning and error message goes to stderr.

Exit codes are stable and map onto the error hierarchy in
:mod:`nanorag.errors` (``docs/cli.md`` has the table)::

    0  success
    1  any other nanorag error, or an unexpected exception
    2  usage error (argparse's own code) — also a missing ingest root
    3  ConfigError   — no generator, missing [local] extra, bad setting
    4  ProviderError — auth, quota, rate limit, retired model, 5xx
    5  StoreError    — index built with another model, no index to inspect
    6  eval only: a metric fell below a committed threshold (report printed)

Keys come from the environment (plan.md §4). ``.env`` in the working
directory is read as a convenience — ``KEY=VALUE`` lines exported only
where the variable is not already set, values never printed — and
``--env-file`` points elsewhere. That is the "three-line optional read",
not a dependency.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from nanorag import __version__
from nanorag.cli import eval as eval_command
from nanorag.cli import ingest, inspect, query, sync
from nanorag.config import Settings
from nanorag.errors import ConfigError, NanoRagError, ProviderError, StoreError

EXIT_OK = 0
EXIT_FAILURE = 1
EXIT_USAGE = 2
EXIT_CONFIG = 3
EXIT_PROVIDER = 4
EXIT_STORE = 5

#: Default ``--env-file``; silently ignored when absent.
DEFAULT_ENV_FILE = ".env"

_COMMANDS = {
    "ingest": ingest,
    "query": query,
    "inspect": inspect,
    "eval": eval_command,
    "sync": sync,
}


def main(argv: Sequence[str] | None = None) -> int:
    """Run the command line and return its exit code.

    Parameters
    ----------
    argv
        Arguments without the program name; ``None`` means ``sys.argv[1:]``.

    Returns
    -------
    int
        One of the ``EXIT_*`` codes above. Usage errors exit through
        ``argparse`` (``SystemExit(2)``) like any other CLI. An exception
        that is not a :class:`~nanorag.errors.NanoRagError` propagates —
        it is a bug, and its traceback is the useful output.

    """
    parser = build_parser()
    args = parser.parse_args(argv)
    _utf8_stdout()
    load_dotenv(Path(args.env_file))
    try:
        settings = _settings(args)
        _configure_logging(args.verbose, settings.log_level)
        payload = _COMMANDS[args.command].run(args, settings)
    except NanoRagError as exc:
        return _fail(exc)
    command = _COMMANDS[args.command]
    if args.json:
        print(json.dumps(payload, indent=2, ensure_ascii=False))
    else:
        print(command.render(payload), end="")
    code: int = command.exit_code(payload)
    return code


def build_parser() -> argparse.ArgumentParser:
    """Build the ``nanorag`` argument parser (also used to render ``--help``)."""
    parser = argparse.ArgumentParser(
        prog="nanorag",
        description="Ingest documents; answer questions with cited context.",
    )
    parser.add_argument("--version", action="version", version=f"nanorag {__version__}")
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument(
        "--persist-dir",
        metavar="DIR",
        help="where the index lives (default: settings.persist_dir, '.nanorag')",
    )
    common.add_argument(
        "--profile",
        metavar="NAME",
        help="a named settings profile, e.g. 'local' (default: $NANORAG_PROFILE)",
    )
    common.add_argument(
        "--embedding-model",
        metavar="MODEL_ID",
        help="fastembed model id (default: settings.embedding_model)",
    )
    common.add_argument(
        "--env-file",
        metavar="PATH",
        default=DEFAULT_ENV_FILE,
        help="KEY=VALUE file exported into the environment where unset "
        f"(default: {DEFAULT_ENV_FILE}; ignored when absent)",
    )
    common.add_argument(
        "--json",
        action="store_true",
        help="print the result as one JSON object (schema: docs/cli.md)",
    )
    common.add_argument(
        "-v",
        "--verbose",
        action="count",
        default=0,
        help="log INFO to stderr; -vv logs DEBUG (default: settings.log_level)",
    )
    subparsers = parser.add_subparsers(dest="command", required=True, metavar="COMMAND")
    for name, module in _COMMANDS.items():
        module.add_parser(subparsers, name, parents=[common])
    return parser


def load_dotenv(path: Path) -> int:
    """Export the ``KEY=VALUE`` lines of *path* that are not already set.

    Returns the number of variables set; ``0`` when *path* does not exist.
    Blank lines, ``#`` comments, a leading ``export`` and matching quotes
    around the value are tolerated. Nothing is ever printed or logged.
    """
    if not path.is_file():
        return 0
    exported = 0
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip().removeprefix("export ").strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        key, value = key.strip(), value.strip()
        if len(value) >= 2 and value[0] == value[-1] and value[0] in "'\"":
            value = value[1:-1]
        if key and key not in os.environ:
            os.environ[key] = value
            exported += 1
    return exported


def _settings(args: argparse.Namespace) -> Settings:
    """Resolve ``Settings`` with the command-line overrides on top."""
    overrides: dict[str, Any] = {}
    if args.persist_dir is not None:
        overrides["persist_dir"] = args.persist_dir
    if args.embedding_model is not None:
        overrides["embedding_model"] = args.embedding_model
    if getattr(args, "generator", None) is not None:
        overrides["generator_preset"] = args.generator
    return Settings.load(profile=args.profile, **overrides)


def _configure_logging(verbose: int, default_level: str) -> None:
    """Send the ``nanorag`` logger to stderr at the requested level."""
    level = {0: default_level, 1: "INFO"}.get(verbose, "DEBUG")
    logging.basicConfig(
        stream=sys.stderr, level=level, format="%(name)s: %(message)s", force=True
    )


def _utf8_stdout() -> None:
    """Never let a console encoding turn an answer into ``UnicodeEncodeError``.

    A real model output can hold characters Windows' default cp1252 console
    cannot encode (seen at Gate C: U+202F from Groq). ``reconfigure`` is
    absent on some redirected/captured streams, in which case they are
    already UTF-8.
    """
    reconfigure = getattr(sys.stdout, "reconfigure", None)
    if reconfigure is not None:
        reconfigure(encoding="utf-8", errors="replace")


def _fail(exc: NanoRagError) -> int:
    """Print *exc* to stderr and return the exit code for its type."""
    print(f"nanorag: error: {exc}", file=sys.stderr)
    if isinstance(exc, ConfigError):
        return EXIT_CONFIG
    if isinstance(exc, ProviderError):
        return EXIT_PROVIDER
    if isinstance(exc, StoreError):
        return EXIT_STORE
    return EXIT_FAILURE
