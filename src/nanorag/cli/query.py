"""``nanorag query QUESTION`` — retrieve, generate, print a cited answer.

The read path exactly as ``Rag.query`` runs it: local query embedding,
exact search over the persisted index, a budgeted nonce-fenced context,
one generation call, resolved citations. The payload's ``answer`` is
``Answer.to_dict()`` — the frozen public type, so its schema is the
library's own contract — and ``sources`` resolves each context block to
the file it came from, which ``Answer`` itself only carries by ``doc_id``.
"""

from __future__ import annotations

import argparse
import json
import logging
from typing import Any

from nanorag.config import Settings
from nanorag.pipeline import DEFAULT_K, NO_PROVIDER, Rag

log = logging.getLogger("nanorag.cli")

_GENERATORS = ("auto", "groq", "gemini", "ollama")


def add_parser(
    subparsers: argparse._SubParsersAction[argparse.ArgumentParser],
    name: str,
    *,
    parents: list[argparse.ArgumentParser],
) -> None:
    """Register the ``query`` subcommand."""
    sub = subparsers.add_parser(
        name,
        parents=parents,
        help="answer QUESTION from the index, with citations",
        description="Answer a question from the index at --persist-dir. "
        "Only the single generation call touches the network.",
    )
    sub.add_argument("question")
    sub.add_argument(
        "-k",
        type=int,
        default=DEFAULT_K,
        help=f"chunks to retrieve (default: {DEFAULT_K})",
    )
    sub.add_argument(
        "--filter",
        type=_json_object,
        metavar="JSON",
        help="metadata pre-filter as a JSON object, e.g. "
        """'{"source_uri": {"$prefix": "docs/"}}'""",
    )
    sub.add_argument(
        "--generator",
        choices=_GENERATORS,
        help="which provider to use; 'auto' chains every available one "
        "(default: settings.generator_preset)",
    )
    sub.set_defaults(parser=sub)


def _json_object(raw: str) -> dict[str, Any]:
    try:
        value = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise argparse.ArgumentTypeError(f"not valid JSON: {exc}") from None
    if not isinstance(value, dict):
        raise argparse.ArgumentTypeError("a filter must be a JSON object")
    return value


def run(args: argparse.Namespace, settings: Settings) -> dict[str, Any]:
    """Answer ``args.question``; return the payload (schema: ``docs/cli.md``)."""
    if args.k < 1:
        args.parser.error(f"-k must be >= 1, got {args.k}")
    rag = Rag.from_defaults(settings=settings, k=args.k)
    try:
        documents = rag.docs.count_documents()
        if documents == 0:
            log.warning(
                "the index at %s holds no documents — run `nanorag ingest ROOT` first",
                settings.persist_dir,
            )
        log.info(
            "index %s: %d documents; k=%d", settings.persist_dir, documents, args.k
        )
        answer = rag.query(args.question, filter=args.filter)
        sources = []
        for label, hit in enumerate(answer.contexts, start=1):
            document = rag.docs.get_document(hit.chunk.doc_id)
            sources.append(
                {
                    "label": label,
                    "score": hit.score,
                    "chunk_id": hit.chunk.chunk_id,
                    "doc_id": hit.chunk.doc_id,
                    "source_uri": document.source_uri if document else None,
                }
            )
    finally:
        rag.close()
    return {
        "question": args.question,
        "k": args.k,
        "filter": args.filter,
        "answer": answer.to_dict(),
        "sources": sources,
    }


def render(payload: dict[str, Any]) -> str:
    """Render a query payload as text: the answer, then its sources."""
    answer = payload["answer"]
    usage, timings = answer["usage"], answer["timings"]
    lines = [answer["text"], ""]
    if usage["provider"] == NO_PROVIDER:
        lines.append("no usable context was retrieved; the model was not called")
    else:
        lines.append(
            f"served by {usage['provider']} ({usage['model']}), "
            f"{usage['total_tokens']} tokens, {timings['total_ms']:.0f} ms"
        )
    flags = []
    if answer["insufficient_context"]:
        flags.append("insufficient context")
    if answer["truncated"]:
        flags.append("context truncated to fit the window")
    if flags:
        lines.append("flags: " + "; ".join(flags))
    if answer["citations"]:
        lines += ["", "cited:"]
        for c in answer["citations"]:
            lines.append(
                f"  [{c['label']}] {c['source_uri']} "
                f"chars {c['start_char']}-{c['end_char']}"
            )
    if payload["sources"]:
        lines += ["", "context blocks in the prompt (label, score, source):"]
        for s in payload["sources"]:
            lines.append(f"  [{s['label']}] {s['score']:.3f}  {s['source_uri']}")
    return "\n".join(lines) + "\n"
