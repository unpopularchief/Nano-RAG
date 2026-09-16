"""``nanorag eval DATASET.jsonl`` — grade the pipeline on a frozen dataset.

By default a **retrieval** run: the corpus is embedded locally into an
in-memory index (the embedding cache under ``--persist-dir`` makes a
repeat, or a chunk-size sweep, cheap), every question is retrieved, and
the ranking is graded against the gold spans. Offline, keyless. With
``--answers`` it is an **answer** run instead: one generation call per
item through the configured generator, graded on abstention, citations
and token F1 — the two-generator comparison plan.md §9 D3 asks for.

``--thresholds FILE`` turns a run into a gate: the report is checked
against the committed baselines and a regression past a tolerance band
exits ``6`` (``EXIT_THRESHOLD``) after printing the report, so CI fails
and the numbers are still in the log.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

from nanorag.chunking.recursive import RecursiveChunker
from nanorag.config import Settings
from nanorag.evaluation.datasets import load_dataset
from nanorag.evaluation.report import (
    EvalReport,
    check_thresholds,
    load_thresholds,
    render_report,
)
from nanorag.evaluation.runner import (
    DEFAULT_KS,
    build_rag,
    evaluate_answers,
    evaluate_retrieval,
)
from nanorag.pipeline import (
    DEFAULT_K,
    default_embedder,
    default_generator,
    default_min_score,
)

#: Exit code for a run whose metrics fell below a committed threshold.
EXIT_THRESHOLD = 6


def add_parser(
    subparsers: argparse._SubParsersAction[argparse.ArgumentParser],
    name: str,
    *,
    parents: list[argparse.ArgumentParser],
) -> None:
    """Register the ``eval`` subcommand."""
    sub = subparsers.add_parser(
        name,
        parents=parents,
        help="grade retrieval (offline) or answers on a dataset; gate on thresholds",
        description="Evaluate the pipeline on DATASET (a dev.jsonl; its corpus/ "
        "sits beside it unless --corpus says otherwise). Retrieval runs are "
        "offline; --answers makes one generation call per item.",
    )
    sub.add_argument("dataset", help="path to the dataset's .jsonl file")
    sub.add_argument(
        "--corpus",
        metavar="DIR",
        help="corpus directory (default: corpus/ next to DATASET)",
    )
    sub.add_argument(
        "--ks",
        default=",".join(str(k) for k in DEFAULT_KS),
        metavar="K,K,...",
        help="cut-offs to report for a retrieval run (default: %(default)s)",
    )
    sub.add_argument(
        "--chunk-tokens",
        type=int,
        default=RecursiveChunker().target_tokens,
        help="RecursiveChunker target size (default: the library default)",
    )
    sub.add_argument(
        "--overlap-tokens",
        type=int,
        default=RecursiveChunker().overlap_tokens,
        help="RecursiveChunker overlap (default: the library default)",
    )
    sub.add_argument(
        "--min-score",
        type=float,
        default=None,
        help="abstention floor; default: the measured floor for the default "
        "embedder, none for any other",
    )
    sub.add_argument(
        "--answers",
        action="store_true",
        help="answer run: call the generator once per item (needs a provider)",
    )
    sub.add_argument(
        "-k",
        type=int,
        default=DEFAULT_K,
        help=f"chunks per query in an answer run (default: {DEFAULT_K})",
    )
    sub.add_argument(
        "--generator",
        choices=("auto", "groq", "gemini", "ollama"),
        help="provider for --answers (default: settings.generator_preset)",
    )
    sub.add_argument(
        "--thresholds",
        metavar="FILE",
        help="check the report against this thresholds.json; exit 6 on a regression",
    )
    sub.set_defaults(parser=sub)


def run(args: argparse.Namespace, settings: Settings) -> dict[str, Any]:
    """Evaluate; return the payload (schema: ``docs/cli.md``)."""
    try:
        ks = tuple(sorted({int(k) for k in args.ks.split(",")}))
    except ValueError:
        args.parser.error(f"--ks must be comma-separated integers, got {args.ks!r}")
    if not ks or min(ks) < 1 or args.k < 1:
        args.parser.error("--ks and -k must be >= 1")
    thresholds = load_thresholds(args.thresholds) if args.thresholds else None
    dataset = load_dataset(args.dataset, args.corpus)
    root = Path(settings.persist_dir)
    root.mkdir(parents=True, exist_ok=True)
    min_score = (
        args.min_score
        if args.min_score is not None
        else default_min_score(settings.embedding_model)
    )
    rag = build_rag(
        dataset,
        embedder=default_embedder(settings, root),
        chunker=RecursiveChunker(
            target_tokens=args.chunk_tokens, overlap_tokens=args.overlap_tokens
        ),
        generator=default_generator(settings) if args.answers else None,
        k=max(ks) if not args.answers else args.k,
        min_score=min_score,
    )
    try:
        report: EvalReport = (
            evaluate_answers(dataset, rag, k=args.k)
            if args.answers
            else evaluate_retrieval(dataset, rag, ks=ks)
        )
    finally:
        rag.close()
    payload: dict[str, Any] = {"report": report.to_dict(), "thresholds": None}
    if thresholds is not None:
        violations = check_thresholds(report, thresholds)
        payload["thresholds"] = {
            "path": args.thresholds,
            "passed": not violations,
            "violations": [v.to_dict() for v in violations],
        }
    return payload


def render(payload: dict[str, Any]) -> str:
    """Render an eval payload as text: the report, then the threshold verdict."""
    data = payload["report"]
    report = EvalReport(
        dataset=data["dataset"],
        kind=data["kind"],
        config=data["config"],
        n_items=data["n_items"],
        n_answerable=data["n_answerable"],
        metrics=data["metrics"],
    )
    text = render_report(report)
    gate = payload["thresholds"]
    if gate is None:
        return text
    if gate["passed"]:
        return text + f"\nthresholds ({gate['path']}): passed\n"
    lines = [f"\nthresholds ({gate['path']}): FAILED"]
    for v in gate["violations"]:
        lines.append(
            f"  {v['metric']}: {v['value']:.4f} < floor {v['floor']:.4f} "
            f"(baseline {v['baseline']:.4f})"
        )
    return text + "\n".join(lines) + "\n"


def exit_code(payload: dict[str, Any]) -> int:
    """Return ``EXIT_THRESHOLD`` when a checked threshold failed, else ``0``."""
    gate = payload["thresholds"]
    return EXIT_THRESHOLD if gate is not None and not gate["passed"] else 0
