"""Run a dataset through the real pipeline and produce an ``EvalReport``.

Two runs, both deterministic given deterministic components:

- :func:`evaluate_retrieval` — needs only an embedder. Ingests the frozen
  corpus into an in-memory index through ``Rag.ingest`` (the same write
  path users run), retrieves the top ``max(ks)`` for every item through
  ``Rag.retrieve``, and grades the ranking against the gold spans with the
  functions in :mod:`~nanorag.evaluation.retrieval_metrics`. Zero model
  calls, zero network: this is the run that gates CI.
- :func:`evaluate_answers` — needs a generator. Runs ``Rag.query`` for
  every item and grades the ``Answer`` with
  :mod:`~nanorag.evaluation.answer_metrics`. One metered call per item;
  meant for the two-generator comparison plan.md §9 D3 asks for (a small
  local model *and* a large API model), never for CI.

The ``Rag`` is built by :func:`build_rag` from the caller's components so
the runner measures exactly the pipeline a user would get — it composes
public pieces and owns no retrieval logic of its own (plan.md §5 rule 1).
"""

from __future__ import annotations

import statistics
from collections.abc import Sequence
from typing import Any

from nanorag.chunking.base import Chunker
from nanorag.embeddings.base import Embedder
from nanorag.errors import AuthError, ProviderError, QuotaExhausted
from nanorag.evaluation.answer_metrics import (
    abstained,
    citation_precision,
    citation_validity,
    token_f1,
)
from nanorag.evaluation.datasets import EvalDataset
from nanorag.evaluation.report import EvalReport
from nanorag.evaluation.retrieval_metrics import (
    Coverage,
    coverage_of,
    hit_at_k,
    ndcg_at_k,
    precision_at_k,
    recall_at_k,
    reciprocal_rank,
)
from nanorag.generation.base import Generator
from nanorag.generation.null import NullGenerator
from nanorag.pipeline import Rag, insufficient_context, rerank_hits
from nanorag.rerank.base import Reranker
from nanorag.store.sqlite_docs import SqliteDocumentStore
from nanorag.tokens import TokenCounter

#: The cut-offs a retrieval report carries by default.
DEFAULT_KS: tuple[int, ...] = (1, 3, 5, 10)
#: An answer run stops after this many items failed back to back — a
#: provider that has rate-limited or broken for ten items in a row is not
#: coming back for the next three hundred.
MAX_CONSECUTIVE_FAILURES = 10


def build_rag(
    dataset: EvalDataset,
    *,
    embedder: Embedder,
    chunker: Chunker | None = None,
    generator: Generator | None = None,
    counter: TokenCounter | None = None,
    k: int = max(DEFAULT_KS),
    min_score: float | None = None,
    reranker: Reranker | None = None,
    retrieve_k: int | None = None,
) -> Rag:
    """Ingest *dataset*'s corpus into a fresh in-memory ``Rag``.

    *generator* defaults to a :class:`~nanorag.generation.NullGenerator` —
    enough for :func:`evaluate_retrieval`; :func:`evaluate_answers` needs a
    real one. *reranker*/*retrieve_k* default to ``Rag``'s own defaults
    (identity, no oversampling) — passed through so the Phase F reranking
    sweep (``benchmarks/rerank_sweep.py``) can measure a real reranker
    against the same harness the committed thresholds were measured with.
    Everything else is passed straight to ``Rag``.
    """
    rag = Rag(
        embedder=embedder,
        generator=generator if generator is not None else NullGenerator(),
        docs=SqliteDocumentStore(":memory:"),
        chunker=chunker,
        counter=counter,
        k=k,
        min_score=min_score,
        reranker=reranker,
        retrieve_k=retrieve_k,
    )
    rag.ingest(dataset.documents)
    return rag


def evaluate_retrieval(
    dataset: EvalDataset, rag: Rag, *, ks: Sequence[int] = DEFAULT_KS
) -> EvalReport:
    """Grade ``rag.retrieve`` on every item at each cut-off in *ks*.

    Answerable items yield ``recall@k`` / ``precision@k`` / ``hit_rate@k``
    for each ``k``, ``mrr``, ``ndcg@k``, and ``false_abstain_rate`` (the
    share ``insufficient_context`` flagged under ``rag``'s thresholds).
    Unanswerable items yield only ``abstain_rate`` (the share flagged).
    Every aggregate is a plain mean over its population.

    Ranking metrics (recall/precision/hit-rate/mrr/ndcg) are graded on
    ``rag``'s full read path including reranking — that is the point of
    measuring a reranker's effect at all. Abstention is graded on the
    retriever's own dense hits, matching ``Rag.query()`` exactly (plan.md
    §18 F10: the floor is calibrated against the embedder's score scale,
    which a reranker's score generally is not).
    """
    if not ks or any(k < 1 for k in ks):
        raise ValueError(f"ks must be non-empty positive integers, got {ks!r}")
    ks = tuple(sorted(set(ks)))
    k_max = max(ks)
    rows: list[dict[str, Any]] = []
    answerable: dict[str, list[float]] = {}
    flagged_answerable: list[float] = []
    flagged_unanswerable: list[float] = []

    for item in dataset.items:
        dense_hits = rag.retriever.retrieve(item.question, k=max(rag.retrieve_k, k_max))
        hits = rerank_hits(rag.reranker, item.question, dense_hits, k_max)
        flagged = insufficient_context(
            dense_hits[:k_max],
            min_results=rag.min_results,
            min_gap=rag.min_gap,
            min_score=rag.min_score,
        )
        row: dict[str, Any] = {
            "id": item.id,
            "answerable": item.answerable,
            "tags": list(item.tags),
            "top_score": hits[0].score if hits else None,
            "flagged_insufficient": flagged,
            "retrieved": [h.chunk.chunk_id for h in hits],
        }
        if not item.answerable:
            flagged_unanswerable.append(1.0 if flagged else 0.0)
            rows.append(row)
            continue

        gold = [span.as_triple() for span in dataset.spans[item.id]]
        ranked: list[Coverage] = [
            coverage_of(h.chunk.doc_id, h.chunk.start_char, h.chunk.end_char, gold)
            for h in hits
        ]
        measured: dict[str, float] = {"mrr": reciprocal_rank(ranked)}
        for k in ks:
            measured[f"recall@{k}"] = recall_at_k(ranked, len(gold), k)
            measured[f"precision@{k}"] = precision_at_k(ranked, k)
            measured[f"hit_rate@{k}"] = hit_at_k(ranked, k)
            measured[f"ndcg@{k}"] = ndcg_at_k(ranked, len(gold), k)
        for name, value in measured.items():
            answerable.setdefault(name, []).append(value)
        flagged_answerable.append(1.0 if flagged else 0.0)
        row["first_relevant_rank"] = next(
            (i for i, c in enumerate(ranked, start=1) if c), None
        )
        row["metrics"] = measured
        rows.append(row)

    metrics = {name: statistics.fmean(values) for name, values in answerable.items()}
    if flagged_answerable:
        metrics["false_abstain_rate"] = statistics.fmean(flagged_answerable)
    if flagged_unanswerable:
        metrics["abstain_rate"] = statistics.fmean(flagged_unanswerable)
    return EvalReport(
        dataset=dataset.name,
        kind="retrieval",
        config={**_pipeline_config(rag), "ks": list(ks)},
        n_items=len(dataset.items),
        n_answerable=len(dataset.answerable),
        metrics=metrics,
        items=tuple(rows),
    )


def evaluate_answers(
    dataset: EvalDataset, rag: Rag, *, k: int | None = None
) -> EvalReport:
    """Grade ``rag.query`` on every item (one generation call each).

    Metrics: ``answer_rate`` (answerable items not abstained on),
    ``abstain_rate`` (unanswerable items abstained on), ``cited_rate``
    (answered answerable items carrying at least one resolved citation),
    ``citation_validity`` and ``citation_precision`` (means over the
    answers where each is defined), ``token_f1`` (mean over items with a
    reference answer), plus ``mean_total_tokens`` and ``mean_generate_ms``.

    A run of hundreds of metered calls must not lose everything to one
    failure at item 80. A ``ProviderError`` on an item is recorded on that
    item's row (``"error"``) and the run continues — unless it is a
    :class:`~nanorag.errors.QuotaExhausted` or
    :class:`~nanorag.errors.AuthError`, after which every remaining call
    would fail the same way, or the :data:`MAX_CONSECUTIVE_FAILURES`-th
    failure in a row (a free tier's per-minute limit surfaces as a
    ``RateLimitError`` on every item once the retries are spent): the run
    **stops** there and the remaining items are recorded as skipped.
    ``completed_rate`` in the metrics says how much of the dataset the
    other numbers are averaged over; a value below ``1.0`` is the first
    thing to read.
    """
    rows: list[dict[str, Any]] = []
    completed: list[float] = []
    answered: list[float] = []
    abstained_on: list[float] = []
    cited: list[float] = []
    validity: list[float] = []
    precision: list[float] = []
    f1: list[float] = []
    tokens: list[float] = []
    generate_ms: list[float] = []

    stopped: str | None = None
    failures_in_a_row = 0
    for item in dataset.items:
        base = {"id": item.id, "answerable": item.answerable, "tags": list(item.tags)}
        if stopped is not None:
            rows.append({**base, "error": f"skipped: run stopped after {stopped}"})
            completed.append(0.0)
            continue
        try:
            answer = rag.query(item.question, k=k)
        except ProviderError as exc:
            rows.append({**base, "error": str(exc)})
            completed.append(0.0)
            failures_in_a_row += 1
            if isinstance(exc, QuotaExhausted | AuthError):
                stopped = type(exc).__name__
            elif failures_in_a_row >= MAX_CONSECUTIVE_FAILURES:
                stopped = f"{failures_in_a_row} consecutive failures"
            continue
        failures_in_a_row = 0
        completed.append(1.0)
        gold = [span.as_triple() for span in dataset.spans[item.id]]
        did_abstain = abstained(answer)
        item_validity = citation_validity(answer)
        item_precision = citation_precision(answer, gold) if gold else None
        item_f1 = token_f1(answer.text, item.answer) if item.answer else None
        if item.answerable:
            answered.append(0.0 if did_abstain else 1.0)
            if not did_abstain:
                cited.append(1.0 if answer.citations else 0.0)
        else:
            abstained_on.append(1.0 if did_abstain else 0.0)
        if item_validity is not None:
            validity.append(item_validity)
        if item_precision is not None:
            precision.append(item_precision)
        if item_f1 is not None:
            f1.append(item_f1)
        tokens.append(answer.usage.total_tokens)
        generate_ms.append(answer.timings.generate_ms)
        rows.append(
            {
                **base,
                "text": answer.text,
                "abstained": did_abstain,
                "insufficient_context": answer.insufficient_context,
                "truncated": answer.truncated,
                "citations": [c.label for c in answer.citations],
                "citation_validity": item_validity,
                "citation_precision": item_precision,
                "token_f1": item_f1,
                "provider": answer.usage.provider,
                "model": answer.usage.model,
                "total_tokens": answer.usage.total_tokens,
                "generate_ms": answer.timings.generate_ms,
            }
        )

    metrics: dict[str, float] = {}
    for name, values in (
        ("completed_rate", completed),
        ("answer_rate", answered),
        ("abstain_rate", abstained_on),
        ("cited_rate", cited),
        ("citation_validity", validity),
        ("citation_precision", precision),
        ("token_f1", f1),
        ("mean_total_tokens", tokens),
        ("mean_generate_ms", generate_ms),
    ):
        if values:
            metrics[name] = statistics.fmean(values)
    generator = rag.generator
    return EvalReport(
        dataset=dataset.name,
        kind="answer",
        config={
            **_pipeline_config(rag),
            "k": k if k is not None else rag.retriever.k,
            "generator": generator.provider,
            "model": generator.model,
        },
        n_items=len(dataset.items),
        n_answerable=len(dataset.answerable),
        metrics=metrics,
        items=tuple(rows),
    )


def _pipeline_config(rag: Rag) -> dict[str, Any]:
    """Return the components a report's numbers depend on, as JSON scalars."""
    chunker = rag.chunker
    return {
        "embedder": rag.embedder.model_id,
        "dim": rag.embedder.dim,
        "chunker": type(chunker).__name__,
        "target_tokens": getattr(chunker, "target_tokens", None),
        "overlap_tokens": getattr(chunker, "overlap_tokens", None),
        "chunks": rag.docs.count_chunks(),
        "min_score": rag.min_score,
        "min_gap": rag.min_gap,
    }
