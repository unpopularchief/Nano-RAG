"""``evaluation/runner.py`` + ``report.py``: the retrieval and answer runs end to
end with fakes (deterministic — two runs produce the same report), the
metrics they aggregate against hand-computed values, the per-item rows, and
the committed-threshold gate."""

import json

import pytest

from nanorag.chunking import FixedChunker
from nanorag.errors import ConfigError, EvaluationError
from nanorag.evaluation import (
    DEFAULT_KS,
    EvalReport,
    Threshold,
    Violation,
    build_rag,
    check_thresholds,
    evaluate_answers,
    evaluate_retrieval,
    load_dataset,
    load_thresholds,
    render_report,
)
from nanorag.prompting import INSUFFICIENT_CONTEXT_TEXT
from nanorag.tokens import HeuristicCounter
from tests.fakes import FakeEmbedder, FakeGenerator

CORPUS = {
    "cats.md": "Cats sleep most of the day.\n\nA cat purrs when it is content.\n",
    "dogs.md": "Dogs bark at strangers.\n\nA dog wags its tail when happy.\n",
    "fish.md": "Fish swim in water.\n\nGoldfish live in bowls.\n",
}

ITEMS = [
    {
        "id": "cats",
        "question": "cats sleep day",
        "gold": [{"source_uri": "cats.md", "quote": "Cats sleep most of the day"}],
        "answer": "most of the day",
    },
    {
        "id": "dogs",
        "question": "dog wags tail happy",
        "gold": [{"source_uri": "dogs.md", "quote": "wags its tail"}],
    },
    {
        "id": "two-spans",
        "question": "purrs content bark strangers",
        "gold": [
            {"source_uri": "cats.md", "quote": "purrs when it is content"},
            {"source_uri": "dogs.md", "quote": "bark at strangers"},
        ],
    },
    {"id": "off-topic", "question": "quantum chromodynamics lattice", "gold": []},
]


@pytest.fixture
def dataset(tmp_path):
    corpus = tmp_path / "corpus"
    corpus.mkdir()
    for name, body in CORPUS.items():
        (corpus / name).write_text(body, encoding="utf-8")
    (tmp_path / "dev.jsonl").write_text(
        "\n".join(json.dumps(i) for i in ITEMS) + "\n", encoding="utf-8"
    )
    return load_dataset(tmp_path / "dev.jsonl")


def _rag(dataset, **kwargs):
    kwargs.setdefault("embedder", FakeEmbedder(dim=256))
    kwargs.setdefault("counter", HeuristicCounter(warn=False))
    kwargs.setdefault(
        "chunker", FixedChunker(target_tokens=8, overlap_tokens=0)
    )  # one sentence per chunk with the heuristic counter
    return build_rag(dataset, **kwargs)


def test_retrieval_run_is_deterministic_and_grades_by_hand(dataset):
    rag = _rag(dataset)
    first = evaluate_retrieval(dataset, rag, ks=(1, 3))
    second = evaluate_retrieval(dataset, _rag(dataset), ks=(1, 3))
    assert first.to_dict() == second.to_dict()

    assert first.kind == "retrieval"
    assert (first.n_items, first.n_answerable) == (4, 3)
    assert first.config["embedder"] == "fake-embedder"
    assert first.config["chunker"] == "FixedChunker"
    assert first.config["target_tokens"] == 8
    assert first.config["ks"] == [1, 3]
    assert first.config["chunks"] == rag.docs.count_chunks() > 3

    rows = {row["id"]: row for row in first.items}
    # The hashing embedder puts the sentence sharing the most words first.
    assert rows["cats"]["first_relevant_rank"] == 1
    assert rows["dogs"]["first_relevant_rank"] == 1
    assert rows["cats"]["metrics"]["recall@1"] == 1.0
    assert rows["two-spans"]["metrics"]["recall@3"] == 1.0
    assert rows["two-spans"]["metrics"]["recall@1"] == 0.5
    assert rows["off-topic"]["answerable"] is False
    assert "metrics" not in rows["off-topic"]
    assert len(rows["off-topic"]["retrieved"]) == 3

    m = first.metrics
    assert set(m) == {
        "mrr",
        "recall@1",
        "precision@1",
        "hit_rate@1",
        "ndcg@1",
        "recall@3",
        "precision@3",
        "hit_rate@3",
        "ndcg@3",
        "false_abstain_rate",
        "abstain_rate",
    }
    assert m["mrr"] == 1.0
    assert m["recall@1"] == pytest.approx((1 + 1 + 0.5) / 3)
    assert m["recall@3"] == 1.0
    assert m["hit_rate@1"] == 1.0
    # No abstention floor on the fake embedder: nothing is flagged.
    assert m["false_abstain_rate"] == 0.0 and m["abstain_rate"] == 0.0


def test_retrieval_run_measures_the_abstention_floor(dataset):
    # A floor high enough to catch the off-topic query but not the on-topic
    # ones shows up as abstain_rate 1.0 / false_abstain_rate 0.0.
    rag = _rag(dataset, min_score=0.1)
    report = evaluate_retrieval(dataset, rag, ks=(1,))
    rows = {row["id"]: row for row in report.items}
    assert rows["off-topic"]["top_score"] < 0.1 < rows["two-spans"]["top_score"]
    assert report.metrics["abstain_rate"] == 1.0
    assert report.metrics["false_abstain_rate"] == 0.0
    assert report.config["min_score"] == 0.1


def test_retrieval_run_over_one_population_omits_the_other_rate(dataset):
    from dataclasses import replace

    only_answerable = replace(dataset, items=dataset.answerable)
    report = evaluate_retrieval(only_answerable, _rag(dataset), ks=(1,))
    assert "false_abstain_rate" in report.metrics
    assert "abstain_rate" not in report.metrics
    only_unanswerable = replace(dataset, items=dataset.unanswerable)
    report = evaluate_retrieval(only_unanswerable, _rag(dataset), ks=(1,))
    assert set(report.metrics) == {"abstain_rate"}


def test_retrieval_run_rejects_bad_ks(dataset):
    rag = _rag(dataset)
    with pytest.raises(ValueError):
        evaluate_retrieval(dataset, rag, ks=())
    with pytest.raises(ValueError):
        evaluate_retrieval(dataset, rag, ks=(0, 3))
    assert evaluate_retrieval(dataset, rag).config["ks"] == list(DEFAULT_KS)


def test_default_generator_is_null_so_a_query_fails_loudly(dataset):
    rag = _rag(dataset)
    with pytest.raises(ConfigError, match="no generator"):
        rag.query("cats sleep day")


def test_answer_run_grades_abstention_citations_and_f1(dataset):
    generator = FakeGenerator(
        [
            "Cats sleep most of the day. [1]",  # cats: cited on gold
            "They wag. [1][7]",  # dogs: one valid, one hallucinated marker
            "Purring and barking. [2]",  # two-spans: valid, but is it on gold?
            INSUFFICIENT_CONTEXT_TEXT,  # off-topic: abstains
        ]
    )
    rag = _rag(dataset, generator=generator, k=3)
    report = evaluate_answers(dataset, rag, k=2)
    assert report.kind == "answer"
    assert report.config["generator"] == "fake"
    assert report.config["model"] == "fake-generator"
    assert report.config["k"] == 2
    rows = {row["id"]: row for row in report.items}
    assert rows["cats"]["citations"] == [1]
    assert rows["cats"]["citation_validity"] == 1.0
    assert rows["cats"]["citation_precision"] == 1.0
    # "cats sleep most of day 1" vs "most of day": 3 shared of 6 predicted,
    # all 3 of the reference — articles and punctuation dropped first.
    assert rows["cats"]["token_f1"] == pytest.approx(2 * (0.5 * 1.0) / (0.5 + 1.0))
    assert rows["dogs"]["citation_validity"] == 0.5
    assert rows["dogs"]["token_f1"] is None  # no reference answer
    assert rows["off-topic"]["abstained"] is True
    assert rows["off-topic"]["citations"] == []
    assert rows["off-topic"]["citation_validity"] is None
    assert rows["off-topic"]["provider"] == "fake"

    m = report.metrics
    assert m["answer_rate"] == 1.0  # all three answerable items answered
    assert m["abstain_rate"] == 1.0  # the unanswerable one abstained
    assert m["cited_rate"] == 1.0
    assert m["citation_validity"] == pytest.approx((1.0 + 0.5 + 1.0) / 3)
    assert 0.0 < m["citation_precision"] <= 1.0
    assert m["token_f1"] == rows["cats"]["token_f1"]
    assert m["mean_total_tokens"] > 0 and m["mean_generate_ms"] >= 0
    assert len(generator.calls) == 4


def test_answer_run_counts_a_wrong_abstention_against_answer_rate(dataset):
    generator = FakeGenerator(
        [INSUFFICIENT_CONTEXT_TEXT, "Wags. [1]", "Both. [1]", "Made-up answer."]
    )
    report = evaluate_answers(dataset, _rag(dataset, generator=generator))
    assert report.metrics["answer_rate"] == pytest.approx(2 / 3)
    assert report.metrics["abstain_rate"] == 0.0


def test_answer_run_records_provider_errors_and_stops_on_quota(dataset):
    from nanorag.errors import QuotaExhausted, TransientError

    generator = FakeGenerator(
        [
            "Cats sleep. [1]",
            TransientError("5xx survived the retries"),  # recorded, run continues
            QuotaExhausted("daily cap"),  # recorded, run stops
            "never reached",
        ]
    )
    report = evaluate_answers(dataset, _rag(dataset, generator=generator))
    rows = {row["id"]: row for row in report.items}
    assert "text" in rows["cats"]
    assert rows["dogs"]["error"] == "5xx survived the retries"
    assert rows["two-spans"]["error"] == "daily cap"
    assert rows["off-topic"]["error"] == "skipped: run stopped after QuotaExhausted"
    assert len(generator.calls) == 3  # the fourth item was never asked
    assert report.metrics["completed_rate"] == 0.25
    assert report.metrics["answer_rate"] == 1.0  # over the one completed item
    assert "abstain_rate" not in report.metrics  # no unanswerable item completed
    assert report.n_items == 4


def test_answer_run_stops_after_too_many_failures_in_a_row(dataset, tmp_path):
    from nanorag.errors import RateLimitError
    from nanorag.evaluation.runner import MAX_CONSECUTIVE_FAILURES

    # A dataset with more items than the failure cap, every call rate-limited.
    corpus = tmp_path / "c"
    corpus.mkdir()
    (corpus / "a.md").write_text("alpha beta gamma delta", encoding="utf-8")
    n = MAX_CONSECUTIVE_FAILURES + 5
    (tmp_path / "dev.jsonl").write_text(
        "\n".join(
            json.dumps({"id": f"q{i}", "question": f"alpha {i}", "gold": []})
            for i in range(n)
        ),
        encoding="utf-8",
    )
    big = load_dataset(tmp_path / "dev.jsonl")
    generator = FakeGenerator([RateLimitError("429") for _ in range(n)])
    report = evaluate_answers(big, _rag(big, generator=generator))
    assert len(generator.calls) == MAX_CONSECUTIVE_FAILURES
    rows = report.items
    assert rows[MAX_CONSECUTIVE_FAILURES - 1]["error"] == "429"
    assert rows[MAX_CONSECUTIVE_FAILURES]["error"].startswith("skipped: run stopped")
    assert report.metrics == {"completed_rate": 0.0}


def test_answer_run_stops_on_a_rejected_key_too(dataset):
    from nanorag.errors import AuthError

    generator = FakeGenerator([AuthError("bad key")])
    report = evaluate_answers(dataset, _rag(dataset, generator=generator))
    assert report.metrics["completed_rate"] == 0.0
    assert len(generator.calls) == 1
    assert set(report.metrics) == {"completed_rate"}


# --- report + thresholds --------------------------------------------------------


def _report(**metrics):
    return EvalReport(
        dataset="toy",
        kind="retrieval",
        config={"embedder": "fake"},
        n_items=4,
        n_answerable=3,
        metrics=metrics,
    )


def test_report_validates_and_renders():
    with pytest.raises(ValueError):
        _report().__class__(
            dataset="x", kind="other", config={}, n_items=1, n_answerable=0, metrics={}
        )
    with pytest.raises(ValueError):
        EvalReport("x", "answer", {}, n_items=1, n_answerable=2, metrics={})
    text = render_report(_report(**{"recall@5": 0.9, "mrr": 0.75}))
    assert "retrieval eval on toy: 4 items (3 answerable, 1 unanswerable)" in text
    assert "embedder: fake" in text
    assert "recall@5  0.9000" in text and "mrr       0.7500" in text
    assert _report().to_dict()["items"] == []


def test_thresholds_gate(tmp_path):
    path = tmp_path / "thresholds.json"
    path.write_text(
        json.dumps(
            {
                "recall@5": {"baseline": 0.90, "tolerance": 0.05},
                "mrr": {"baseline": 0.80, "tolerance": 0.10},
            }
        ),
        encoding="utf-8",
    )
    thresholds = load_thresholds(path)
    assert thresholds[0] == Threshold("recall@5", 0.90, 0.05)
    assert thresholds[0].floor == 0.85 and thresholds[1].floor == 0.7  # no 0.7000…1

    passing = _report(**{"recall@5": 0.86, "mrr": 0.70})
    assert check_thresholds(passing, thresholds) == ()
    (violation,) = check_thresholds(
        _report(**{"recall@5": 0.84, "mrr": 0.70}), thresholds
    )
    assert violation == Violation("recall@5", 0.84, 0.85, 0.90)
    assert set(violation.to_dict()) == {"metric", "value", "floor", "baseline"}
    with pytest.raises(EvaluationError, match="not in the report"):
        check_thresholds(_report(**{"recall@5": 0.9}), thresholds)


@pytest.mark.parametrize(
    "body, message",
    [
        ("{not json", "not valid JSON"),
        ("[]", "non-empty JSON object"),
        ("{}", "non-empty JSON object"),
        ('{"recall@5": {"baseline": 0.9}}', "'tolerance'"),
        ('{"recall@5": {"baseline": 1.5, "tolerance": 0.1}}', "must be in"),
        ('{"recall@5": {"baseline": "x", "tolerance": 0.1}}", ', "not valid JSON"),
    ],
)
def test_malformed_thresholds_are_rejected(tmp_path, body, message):
    path = tmp_path / "t.json"
    path.write_text(body, encoding="utf-8")
    with pytest.raises(EvaluationError, match=message):
        load_thresholds(path)
    with pytest.raises(EvaluationError, match="no such thresholds file"):
        load_thresholds(tmp_path / "absent.json")


def test_threshold_value_object_validates():
    with pytest.raises(ValueError):
        Threshold("", 0.5, 0.1)
    with pytest.raises(ValueError):
        Threshold("m", 0.5, -0.1)
