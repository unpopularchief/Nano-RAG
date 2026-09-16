"""``cli/eval.py``: the eval subcommand end to end with a stubbed embedder — the
payload schema, the threshold gate's exit code, the answer run through the
default generator, and the usage/eval error paths."""

import json

import pytest

from nanorag.cli import eval as eval_command
from nanorag.cli import main
from nanorag.cli.app import EXIT_FAILURE, EXIT_OK, EXIT_USAGE
from nanorag.cli.eval import EXIT_THRESHOLD
from nanorag.embeddings import local
from tests.conftest import blocked_sockets
from tests.fakes import FakeEmbedder, FakeGenerator
from tests.test_evaluation_runner import CORPUS, ITEMS


class StubLocal(FakeEmbedder):
    def __init__(self, model_id):
        super().__init__(dim=256, model_id=model_id)


@pytest.fixture
def dataset_path(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    for var in ("NANORAG_PROFILE", "NANORAG_PERSIST_DIR", "NANORAG_GENERATOR_PRESET"):
        monkeypatch.delenv(var, raising=False)
    monkeypatch.setattr(local, "FastEmbedEmbedder", StubLocal)
    root = tmp_path / "toy"
    (root / "corpus").mkdir(parents=True)
    for name, body in CORPUS.items():
        (root / "corpus" / name).write_text(body, encoding="utf-8")
    (root / "dev.jsonl").write_text(
        "\n".join(json.dumps(i) for i in ITEMS) + "\n", encoding="utf-8"
    )
    return root / "dev.jsonl"


def _thresholds(path, **entries):
    path.write_text(
        json.dumps(
            {m: {"baseline": b, "tolerance": t} for m, (b, t) in entries.items()}
        ),
        encoding="utf-8",
    )
    return str(path)


def test_retrieval_eval_json_payload_and_stays_offline(dataset_path, capsys):
    with blocked_sockets():  # a retrieval run must never touch the network
        code = main(
            [
                "eval",
                str(dataset_path),
                "--json",
                "--persist-dir",
                "cache",
                "--embedding-model",
                "stub",
                "--ks",
                "1,3",
                "--chunk-tokens",
                "32",
                "--overlap-tokens",
                "0",
            ]
        )
    assert code == EXIT_OK
    out, err = capsys.readouterr()
    assert err == ""
    payload = json.loads(out)
    assert set(payload) == {"report", "thresholds"}
    assert payload["thresholds"] is None
    report = payload["report"]
    assert set(report) == {
        "dataset",
        "kind",
        "config",
        "n_items",
        "n_answerable",
        "metrics",
        "items",
    }
    assert report["dataset"] == "toy" and report["kind"] == "retrieval"
    assert (report["n_items"], report["n_answerable"]) == (4, 3)
    assert report["config"]["embedder"] == "stub"
    assert report["config"]["target_tokens"] == 32
    assert report["config"]["overlap_tokens"] == 0
    assert report["config"]["ks"] == [1, 3]
    assert report["config"]["min_score"] is None  # not the default embedder
    assert {"recall@1", "recall@3", "mrr", "ndcg@3"} <= set(report["metrics"])
    assert len(report["items"]) == 4
    # the embedding cache lives under --persist-dir; no index is written there
    assert (dataset_path.parent.parent / "cache" / "embeddings.sqlite").is_file()
    assert not (dataset_path.parent.parent / "cache" / "nanorag.sqlite").exists()


def test_default_embedder_gets_the_measured_floor_and_min_score_overrides(
    dataset_path, capsys
):
    main(["eval", str(dataset_path), "--json", "--persist-dir", "c"])
    assert json.loads(capsys.readouterr().out)["report"]["config"]["min_score"] == 0.55
    main(
        ["eval", str(dataset_path), "--json", "--persist-dir", "c", "--min-score", "0"]
    )
    assert json.loads(capsys.readouterr().out)["report"]["config"]["min_score"] == 0


def test_threshold_gate_passes_and_fails_with_exit_6(dataset_path, capsys, tmp_path):
    base = ["eval", str(dataset_path), "--persist-dir", "c", "--embedding-model", "s"]
    assert main(base) == EXIT_OK  # no thresholds: the report alone, no verdict line
    out = capsys.readouterr().out
    assert "recall@3" in out and "thresholds" not in out

    ok = _thresholds(tmp_path / "ok.json", **{"recall@3": (0.9, 0.1)})
    assert main([*base, "--thresholds", ok]) == EXIT_OK
    out = capsys.readouterr().out
    assert "retrieval eval on toy: 4 items (3 answerable, 1 unanswerable)" in out
    assert "thresholds (" in out and "): passed" in out

    bad = _thresholds(tmp_path / "bad.json", **{"recall@1": (1.0, 0.0)})
    assert main([*base, "--thresholds", bad]) == EXIT_THRESHOLD
    out = capsys.readouterr().out
    assert "): FAILED" in out and "recall@1: 0.8333 < floor 1.0000" in out

    assert main([*base, "--thresholds", bad, "--json"]) == EXIT_THRESHOLD
    gate = json.loads(capsys.readouterr().out)["thresholds"]
    assert gate["passed"] is False
    assert gate["violations"][0]["metric"] == "recall@1"
    assert set(gate["violations"][0]) == {"metric", "value", "floor", "baseline"}


def test_answer_run_uses_the_default_generator(dataset_path, capsys, monkeypatch):
    generator = FakeGenerator(["Cats sleep most of the day. [1]"])
    seen = []

    def pick(settings, **_):
        seen.append(settings.generator_preset)
        return generator

    monkeypatch.setattr(eval_command, "default_generator", pick)
    code = main(
        [
            "eval",
            str(dataset_path),
            "--answers",
            "--generator",
            "ollama",
            "-k",
            "2",
            "--json",
            "--persist-dir",
            "c",
            "--embedding-model",
            "s",
        ]
    )
    assert code == EXIT_OK
    report = json.loads(capsys.readouterr().out)["report"]
    assert report["kind"] == "answer"
    assert report["config"]["generator"] == "fake" and report["config"]["k"] == 2
    assert seen == ["ollama"]
    assert len(generator.calls) == 4
    assert {"answer_rate", "abstain_rate", "citation_validity"} <= set(
        report["metrics"]
    )


def test_eval_usage_and_dataset_errors(dataset_path, capsys, tmp_path):
    for argv in (
        ["eval", str(dataset_path), "--ks", "1,x"],
        ["eval", str(dataset_path), "--ks", "0"],
        ["eval", str(dataset_path), "-k", "0"],
        ["eval"],
    ):
        with pytest.raises(SystemExit) as info:
            main(argv)
        assert info.value.code == EXIT_USAGE, argv

    assert main(["eval", str(tmp_path / "absent.jsonl")]) == EXIT_FAILURE
    assert "no such dataset file" in capsys.readouterr().err
    assert main(["eval", str(dataset_path), "--thresholds", "nope.json"]) == 1
    assert "no such thresholds file" in capsys.readouterr().err
    assert main(["eval", str(dataset_path), "--corpus", str(tmp_path / "x")]) == 1
    assert "no such corpus directory" in capsys.readouterr().err
