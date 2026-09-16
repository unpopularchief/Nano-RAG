"""Evaluation: quality as numbers (Phase D, session D3).

``datasets`` (a frozen corpus + graded questions with quote-located gold),
``retrieval_metrics`` (Recall/Precision/hit-rate@k, MRR, nDCG@k),
``answer_metrics`` (citation validity/precision, abstention, token F1),
``runner`` (``evaluate_retrieval`` offline; ``evaluate_answers`` with a
generator) and ``report`` (``EvalReport`` + the committed-threshold gate).
Nothing here imports a provider or a model runtime.
"""

from __future__ import annotations

from nanorag.evaluation.datasets import (
    EvalDataset,
    EvalItem,
    GoldSpan,
    ResolvedSpan,
    load_corpus,
    load_dataset,
    load_items,
    resolve_spans,
)
from nanorag.evaluation.report import (
    EvalReport,
    Threshold,
    Violation,
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

__all__ = [
    "EvalDataset",
    "EvalItem",
    "GoldSpan",
    "ResolvedSpan",
    "load_corpus",
    "load_dataset",
    "load_items",
    "resolve_spans",
    "EvalReport",
    "Threshold",
    "Violation",
    "check_thresholds",
    "load_thresholds",
    "render_report",
    "DEFAULT_KS",
    "build_rag",
    "evaluate_answers",
    "evaluate_retrieval",
]
