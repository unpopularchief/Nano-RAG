"""The committed dataset ``datasets/nanorag-docs`` is well-formed (default
suite, offline, no model): every gold quote resolves uniquely, the set is
large enough to be a regression gate (plan.md §18 F3), the committed
thresholds name metrics the runner actually produces, and every vendored
third-party document is attributed in the dataset README."""

import re
from pathlib import Path

from nanorag.evaluation import (
    build_rag,
    check_thresholds,
    evaluate_retrieval,
    load_dataset,
    load_thresholds,
)
from nanorag.tokens import HeuristicCounter
from tests.fakes import FakeEmbedder

DATASET = Path(__file__).resolve().parents[1] / "datasets" / "nanorag-docs"


def test_dataset_loads_and_is_large_enough_to_gate():
    dataset = load_dataset(DATASET / "dev.jsonl")
    assert len(dataset.answerable) >= 200
    assert len(dataset.unanswerable) >= 20
    assert len(dataset.documents) >= 15
    # every answerable item resolved to at least one span inside its document
    for item in dataset.answerable:
        for span in dataset.spans[item.id]:
            document = next(d for d in dataset.documents if d.doc_id == span.doc_id)
            assert 0 <= span.start_char < span.end_char <= len(document.text)
    # every document is the target of at least one question
    cited = {span.source_uri for spans in dataset.spans.values() for span in spans}
    assert cited == {d.source_uri for d in dataset.documents}


def test_committed_thresholds_match_the_runner_and_the_gate_logic():
    dataset = load_dataset(DATASET / "dev.jsonl")
    thresholds = load_thresholds(DATASET / "thresholds.json")
    assert {t.metric for t in thresholds} >= {"recall@5"}
    for threshold in thresholds:
        assert threshold.tolerance >= 0.025  # at least the noise floor at n≈300
    # A fake embedder scores far below the real baseline; the point here is
    # that every thresholded metric exists in a report from the real runner.
    rag = build_rag(
        dataset, embedder=FakeEmbedder(dim=256), counter=HeuristicCounter(warn=False)
    )
    report = evaluate_retrieval(dataset, rag)
    rag.close()
    assert {t.metric for t in thresholds} <= set(report.metrics)
    assert check_thresholds(report, thresholds)  # the gate does trip on garbage


def test_every_third_party_document_is_attributed():
    readme = (DATASET / "README.md").read_text(encoding="utf-8")
    for path in sorted((DATASET / "corpus" / "third-party").glob("*.md")):
        assert re.search(rf"`{re.escape(path.name)}`", readme), path.name
