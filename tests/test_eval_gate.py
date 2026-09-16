"""The Phase D regression gate (``-m eval``): the committed dataset through the
real embedder, checked against the committed baselines.

Opt-in and slow (a few minutes: the corpus is embedded once per run, every
question is embedded at query time), offline after the model download, and
meant for **one fixed runner image** (``.github/workflows/eval.yml``), not
the CI matrix — ONNX Runtime drifts slightly across builds, which is what
the tolerance band in ``thresholds.json`` absorbs (plan.md §18 F3, F5).

Run locally::

    uv sync --extra dev --extra local
    uv run pytest -q -m eval -s      # -s to see the report

Set ``NANORAG_EVAL_CACHE=<dir>`` to reuse an embedding cache between local
runs (the corpus is then embedded once; queries always are). CI does not
set it — a cached vector would hide exactly the build drift the gate
watches for.

The dataset's own validity (every quote resolves, ≥ 200 graded items) is
checked in the default suite by ``test_evaluation_dataset_nanorag_docs.py``;
this file only measures.
"""

import os
from pathlib import Path

import pytest

from nanorag.config import Settings
from nanorag.evaluation import (
    build_rag,
    check_thresholds,
    evaluate_retrieval,
    load_dataset,
    load_thresholds,
    render_report,
)
from nanorag.pipeline import default_embedder, default_min_score
from tests.conftest import blocked_sockets

pytestmark = pytest.mark.eval

DATASET = Path(__file__).resolve().parents[1] / "datasets" / "nanorag-docs"


def test_committed_dataset_meets_committed_thresholds(tmp_path):
    dataset = load_dataset(DATASET / "dev.jsonl")
    thresholds = load_thresholds(DATASET / "thresholds.json")
    settings = Settings()
    cache = Path(os.environ.get("NANORAG_EVAL_CACHE", tmp_path))
    embedder = default_embedder(settings, cache)  # warms the model outside the block
    with blocked_sockets():  # the gate is offline: no network, ever
        rag = build_rag(
            dataset,
            embedder=embedder,
            min_score=default_min_score(settings.embedding_model),
        )
        try:
            report = evaluate_retrieval(dataset, rag)

            # plan.md §9 Phase E Acceptance: "Phase D eval numbers unchanged
            # by a re-sync." Reuses this same already-embedded `rag` rather
            # than building a second one -- `datasets/nanorag-docs/corpus/`
            # is a real directory that `load_corpus` already walked with
            # `DirectoryLoader` and the default "**/*" glob, the same
            # combination `sync_path` uses, so resyncing it unedited here
            # and re-measuring is the literal acceptance scenario, on the
            # real embedder and the real committed dataset, without paying
            # for a second cold corpus embedding pass.
            sync = rag.sync_path(DATASET / "corpus", glob="**/*", apply=True)
            assert (sync.added, sync.updated, sync.deleted) == ((), (), ())
            resynced = evaluate_retrieval(dataset, rag)
            assert resynced.metrics == report.metrics
        finally:
            rag.close()
    print("\n" + render_report(report))
    violations = check_thresholds(report, thresholds)
    assert not violations, "\n".join(
        f"{v.metric}: {v.value:.4f} < floor {v.floor:.4f} (baseline {v.baseline:.4f})"
        for v in violations
    )
