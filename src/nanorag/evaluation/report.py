"""``EvalReport`` — what a run measured — and the committed-threshold check.

A report is data: the configuration it was measured under, the aggregate
metrics, and one row per item so a regression can be traced to the
questions that moved. ``to_dict()`` is the JSON the CLI prints;
``render_report`` is a text view of the same numbers.

Thresholds (``thresholds.json``) are the regression gate plan.md §9 D3 and
§18 F3 call for: each entry is a metric name, the **baseline** value that
was measured when the file was committed, and a **tolerance** — the band
below the baseline that is not a regression. The band must be at least
the dataset's noise floor (the standard error of a proportion at its size;
~2.5 pp at n = 200, p = 0.85), else the gate trips on noise. A metric is
violated when ``value < baseline − tolerance``. Nothing here decides the
tolerance; the dataset's README records how each one was chosen.

Schema of ``thresholds.json``::

    {"recall@5": {"baseline": 0.91, "tolerance": 0.05},
     "mrr":      {"baseline": 0.78, "tolerance": 0.06}}
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from nanorag.errors import EvaluationError


@dataclass(frozen=True, slots=True)
class EvalReport:
    """The outcome of one evaluation run.

    Attributes
    ----------
    dataset
        The dataset name.
    kind
        ``"retrieval"`` or ``"answer"``.
    config
        Everything the numbers depend on: embedder, chunker, ``k``, the
        abstention floor, the generator for an answer run.
    n_items, n_answerable
        Item counts; ``n_items − n_answerable`` were unanswerable.
    metrics
        Aggregate metric name → value. Names are stable (``recall@5``,
        ``precision@5``, ``hit_rate@5``, ``mrr``, ``ndcg@5``,
        ``abstain_rate``, ``false_abstain_rate``, …).
    items
        One JSON-serialisable row per item, in dataset order.

    """

    dataset: str
    kind: str
    config: Mapping[str, Any]
    n_items: int
    n_answerable: int
    metrics: Mapping[str, float]
    items: tuple[Mapping[str, Any], ...] = field(default_factory=tuple)

    def __post_init__(self) -> None:
        """Validate the field values (see the class docstring)."""
        if self.kind not in ("retrieval", "answer"):
            raise ValueError(
                f"EvalReport.kind must be retrieval|answer, got {self.kind!r}"
            )
        if self.n_items < 0 or not 0 <= self.n_answerable <= self.n_items:
            raise ValueError("EvalReport item counts are inconsistent")

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serialisable dict of this report."""
        return {
            "dataset": self.dataset,
            "kind": self.kind,
            "config": dict(self.config),
            "n_items": self.n_items,
            "n_answerable": self.n_answerable,
            "metrics": dict(self.metrics),
            "items": [dict(row) for row in self.items],
        }


def render_report(report: EvalReport) -> str:
    """Render a report's header and metrics as text (no per-item rows)."""
    lines = [
        f"{report.kind} eval on {report.dataset}: {report.n_items} items "
        f"({report.n_answerable} answerable, "
        f"{report.n_items - report.n_answerable} unanswerable)",
    ]
    lines += [f"  {key}: {value}" for key, value in report.config.items()]
    lines.append("")
    width = max((len(name) for name in report.metrics), default=0)
    lines += [
        f"  {name:<{width}}  {value:.4f}" for name, value in report.metrics.items()
    ]
    return "\n".join(lines) + "\n"


@dataclass(frozen=True, slots=True)
class Threshold:
    """A committed baseline for one metric and the band that is not a regression."""

    metric: str
    baseline: float
    tolerance: float

    def __post_init__(self) -> None:
        """Validate the field values (see the class docstring)."""
        if not self.metric:
            raise ValueError("Threshold.metric must be a non-empty string")
        if not 0.0 <= self.baseline <= 1.0:
            raise ValueError(
                f"Threshold.baseline must be in [0, 1], got {self.baseline}"
            )
        if not 0.0 <= self.tolerance <= 1.0:
            raise ValueError(
                f"Threshold.tolerance must be in [0, 1], got {self.tolerance}"
            )

    @property
    def floor(self) -> float:
        """The lowest value that is not a regression.

        Rounded to nine decimals so ``0.8 − 0.1`` is ``0.7``, not
        ``0.7000000000000001`` — a gate must not trip on float arithmetic.
        """
        return round(self.baseline - self.tolerance, 9)


@dataclass(frozen=True, slots=True)
class Violation:
    """A metric that fell below its threshold's floor."""

    metric: str
    value: float
    floor: float
    baseline: float

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serialisable dict of this violation."""
        return {
            "metric": self.metric,
            "value": self.value,
            "floor": self.floor,
            "baseline": self.baseline,
        }


def load_thresholds(path: str | Path) -> tuple[Threshold, ...]:
    """Read a ``thresholds.json`` (schema in the module docstring).

    Raises
    ------
    EvaluationError
        The file is missing or an entry is malformed.

    """
    path = Path(path)
    if not path.is_file():
        raise EvaluationError(f"no such thresholds file: {path}", path=str(path))
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise EvaluationError(f"{path.name}: not valid JSON ({exc.msg})") from None
    if not isinstance(data, dict) or not data:
        raise EvaluationError(f"{path.name}: expected a non-empty JSON object")
    thresholds = []
    for metric, entry in data.items():
        try:
            thresholds.append(
                Threshold(
                    metric=str(metric),
                    baseline=float(entry["baseline"]),
                    tolerance=float(entry["tolerance"]),
                )
            )
        except (KeyError, TypeError, ValueError) as exc:
            raise EvaluationError(f"{path.name}: {metric!r}: {exc}") from None
    return tuple(thresholds)


def check_thresholds(
    report: EvalReport, thresholds: tuple[Threshold, ...]
) -> tuple[Violation, ...]:
    """Return every threshold *report* falls below; empty means it passed.

    Raises
    ------
    EvaluationError
        A threshold names a metric the report does not have — a silently
        skipped gate is no gate.

    """
    violations = []
    for threshold in thresholds:
        if threshold.metric not in report.metrics:
            raise EvaluationError(
                f"threshold metric {threshold.metric!r} is not in the report",
                available=sorted(report.metrics),
            )
        value = report.metrics[threshold.metric]
        if value < threshold.floor:
            violations.append(
                Violation(threshold.metric, value, threshold.floor, threshold.baseline)
            )
    return tuple(violations)
