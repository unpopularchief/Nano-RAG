"""``cli``: the three subcommands end to end with fakes (no model, key or
network), the ``--json`` payload schemas documented in ``docs/cli.md``,
stdout/stderr separation, exit codes for every mapped error type, the
``.env`` read, and the ``python -m nanorag.cli`` entry point."""

import io
import json
import subprocess
import sys
from pathlib import Path

import pytest

from nanorag import pipeline
from nanorag.cli import main
from nanorag.cli.app import (
    EXIT_CONFIG,
    EXIT_FAILURE,
    EXIT_OK,
    EXIT_PROVIDER,
    EXIT_STORE,
    EXIT_USAGE,
    _utf8_stdout,
    load_dotenv,
)
from nanorag.embeddings import local
from nanorag.errors import (
    AuthError,
    ConfigError,
    QuotaExhausted,
    RetrievalError,
)
from nanorag.pipeline import NO_PROVIDER
from nanorag.prompting import INSUFFICIENT_CONTEXT_TEXT
from tests.fakes import FakeEmbedder, FakeGenerator

FIXTURE_CORPUS = Path(__file__).parent / "data" / "corpus"

CORPUS = {
    "cats.txt": "Cats sleep most of the day. A cat purrs when content.",
    "dogs.txt": "Dogs bark at strangers. A dog wags its tail when happy.",
    "notes/fish.md": "# Fish\n\nFish swim in water. Goldfish live in bowls.",
    "image.png": b"\x89PNG not text",
}

ANSWER = "Cats sleep most of the day. [1]"

ANSWER_KEYS = {
    "text",
    "citations",
    "contexts",
    "insufficient_context",
    "truncated",
    "usage",
    "timings",
}


class StubLocal(FakeEmbedder):
    """Stands in for ``FastEmbedEmbedder`` so ``from_defaults`` needs no model."""

    def __init__(self, model_id):
        super().__init__(dim=64, model_id=model_id)


@pytest.fixture
def cwd(tmp_path, monkeypatch):
    # An isolated working directory: no project pyproject, and — importantly —
    # no real `.env` for the default `--env-file` to pick up.
    monkeypatch.chdir(tmp_path)
    for var in ("NANORAG_PROFILE", "NANORAG_PERSIST_DIR", "NANORAG_GENERATOR_PRESET"):
        monkeypatch.delenv(var, raising=False)
    return tmp_path


@pytest.fixture
def corpus(cwd):
    root = cwd / "corpus"
    for name, body in CORPUS.items():
        path = root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        if isinstance(body, bytes):
            path.write_bytes(body)
        else:
            path.write_text(body, encoding="utf-8")
    return root


@pytest.fixture
def fake_generator(monkeypatch):
    # The generator `from_defaults` would pick; `calls` records the settings
    # it was asked with so the `--generator` mapping can be asserted.
    generator = FakeGenerator([ANSWER])
    generator.settings_seen = []

    def pick(settings, **_):
        generator.settings_seen.append(settings)
        return generator

    monkeypatch.setattr(local, "FastEmbedEmbedder", StubLocal)
    monkeypatch.setattr(pipeline, "default_generator", pick)
    return generator


# A non-default embedding model id, so `from_defaults` does not apply the
# `min_score` floor measured for bge-base — the hashing fake's scores sit
# well under it and every answer would be flagged "insufficient context".
MODEL = ["--embedding-model", "fake-model"]


def _ingest(corpus, *extra):
    return main(["ingest", str(corpus), "--persist-dir", "idx", *MODEL, *extra])


def _query(question, *extra):
    return main(["query", question, "--persist-dir", "idx", *MODEL, *extra])


# --- ingest -------------------------------------------------------------------


def test_ingest_is_offline_and_never_needs_a_generator(
    corpus, fake_generator, monkeypatch, capsys
):
    def refuse(*_, **__):
        raise AssertionError("ingest must not pick a generator")

    monkeypatch.setattr(pipeline, "default_generator", refuse)
    assert _ingest(corpus) == EXIT_OK
    out, err = capsys.readouterr()
    assert out.startswith("ingested 3 documents (3 chunks)")
    assert "skipped 1, failed 0" in out
    assert err == ""
    assert (corpus.parent / "idx" / "nanorag.sqlite").is_file()
    assert (corpus.parent / "idx" / "embeddings.sqlite").is_file()


def test_ingest_json_payload_schema(corpus, fake_generator, capsys):
    assert _ingest(corpus, "--json", "--glob", "**/*.txt", "--ignore", "dogs*") == 0
    payload = json.loads(capsys.readouterr().out)
    assert set(payload) == {
        "persist_dir",
        "root",
        "glob",
        "ignore",
        "chunks",
        "loaded",
        "skipped",
        "failed",
    }
    assert payload["persist_dir"] == "idx"
    assert payload["glob"] == "**/*.txt"
    assert payload["ignore"] == ["dogs*"]
    assert [d["source_uri"] for d in payload["loaded"]] == ["cats.txt"]
    assert set(payload["loaded"][0]) == {"doc_id", "source_uri"}
    assert payload["chunks"] == 1
    assert payload["skipped"] == []  # image.png never matched the glob
    assert payload["failed"] == []


def test_ingest_reports_a_failed_file_and_still_exits_zero(cwd, fake_generator, capsys):
    assert _ingest(FIXTURE_CORPUS, "--json") == EXIT_OK
    payload = json.loads(capsys.readouterr().out)
    assert "corrupt.txt" in {i["source_uri"] for i in payload["failed"]}
    assert all(set(i) == {"source_uri", "reason"} for i in payload["failed"])
    assert len(payload["loaded"]) >= 5

    assert _ingest(FIXTURE_CORPUS) == EXIT_OK
    out = capsys.readouterr().out
    assert "failed 1" in out.splitlines()[0]
    assert "  failed  corrupt.txt:" in out


def test_ingest_root_that_is_not_a_directory_is_a_usage_error(
    cwd, fake_generator, capsys
):
    with pytest.raises(SystemExit) as info:
        _ingest(cwd / "missing")
    assert info.value.code == EXIT_USAGE
    assert "not a directory" in capsys.readouterr().err


# --- query --------------------------------------------------------------------


def test_query_text_output_answer_citations_and_sources(corpus, fake_generator, capsys):
    _ingest(corpus)
    capsys.readouterr()
    assert _query("Do cats sleep?", "-k", "2") == EXIT_OK
    out, err = capsys.readouterr()
    assert out.splitlines()[0] == ANSWER
    assert "served by fake (fake-generator)" in out
    assert "cited:" in out and "[1] cats.txt chars 0-" in out
    assert "context blocks in the prompt" in out
    assert "[1] " in out and "cats.txt" in out
    assert "flags:" not in out
    assert err == ""
    assert len(fake_generator.calls) == 1


def test_query_json_payload_is_answer_to_dict_plus_sources(
    corpus, fake_generator, capsys
):
    _ingest(corpus)
    capsys.readouterr()
    filt = json.dumps({"source_uri": {"$prefix": "cats"}})
    assert _query("Do cats sleep?", "--json", "--filter", filt) == EXIT_OK
    out, err = capsys.readouterr()
    assert err == ""
    payload = json.loads(out)
    assert set(payload) == {"question", "k", "filter", "answer", "sources"}
    assert payload["question"] == "Do cats sleep?"
    assert payload["k"] == 8
    assert payload["filter"] == {"source_uri": {"$prefix": "cats"}}
    answer = payload["answer"]
    assert set(answer) == ANSWER_KEYS
    assert answer["text"] == ANSWER
    assert answer["insufficient_context"] is False
    assert answer["usage"]["provider"] == "fake"
    assert [c["label"] for c in answer["citations"]] == [1]
    assert answer["citations"][0]["source_uri"] == "cats.txt"
    # the filter narrowed retrieval to the one matching document
    assert [s["source_uri"] for s in payload["sources"]] == ["cats.txt"]
    assert set(payload["sources"][0]) == {
        "label",
        "score",
        "chunk_id",
        "doc_id",
        "source_uri",
    }
    first_chunk = answer["contexts"][0]["chunk"]
    assert payload["sources"][0]["chunk_id"] == first_chunk["chunk_id"]


def test_query_on_an_empty_index_abstains_without_a_model_call_and_warns(
    cwd, fake_generator, capsys
):
    assert _query("anything?", "--json") == EXIT_OK
    out, err = capsys.readouterr()
    payload = json.loads(out)
    assert payload["answer"]["text"] == INSUFFICIENT_CONTEXT_TEXT
    assert payload["answer"]["usage"]["provider"] == NO_PROVIDER
    assert payload["sources"] == []
    assert "holds no documents" in err and "nanorag ingest" in err
    assert fake_generator.calls == []

    _query("anything?")
    out, _ = capsys.readouterr()
    assert "the model was not called" in out
    assert "flags: insufficient context" in out


def test_query_reports_truncation_when_nothing_fits_the_window(
    corpus, fake_generator, capsys
):
    _ingest(corpus)
    capsys.readouterr()
    fake_generator.context_window = 300  # the budget floors at 1 token
    fake_generator.max_output_tokens = 256
    assert _query("Do cats sleep?") == EXIT_OK
    out = capsys.readouterr().out
    assert "flags: insufficient context; context truncated to fit the window" in out
    assert "the model was not called" in out
    assert fake_generator.calls == []


def test_query_json_keeps_non_ascii_and_stdout_holds_only_json(
    corpus, fake_generator, capsys
):
    _ingest(corpus)
    capsys.readouterr()
    fake_generator._responses[:] = ["Cats sleep \u2014 mostly. \u202f[1]"]
    assert _query("q", "--json", "-v") == EXIT_OK
    out, err = capsys.readouterr()
    payload = json.loads(out)  # nothing but the JSON object on stdout
    assert "\u202f" in payload["answer"]["text"] and "\\u202f" not in out
    assert "nanorag.cli: index idx: 3 documents; k=8" in err  # -v: INFO on stderr


def test_query_usage_errors(corpus, fake_generator, capsys):
    for argv in (
        ["query", "q", "-k", "0"],
        ["query", "q", "--filter", "{not json"],
        ["query", "q", "--filter", "[1, 2]"],
        ["query", "q", "--generator", "openai"],
        ["query"],
        [],
    ):
        with pytest.raises(SystemExit) as info:
            main(argv)
        assert info.value.code == EXIT_USAGE, argv


def test_query_generator_flag_becomes_the_settings_preset(
    corpus, fake_generator, capsys
):
    _ingest(corpus)
    _query("q", "--generator", "ollama")
    _query("q")
    presets = [s.generator_preset for s in fake_generator.settings_seen]
    assert presets == ["ollama", "auto"]
    assert all(s.persist_dir == "idx" for s in fake_generator.settings_seen)


def test_profile_and_embedding_model_flags_reach_settings(
    corpus, fake_generator, capsys
):
    assert main(["ingest", str(corpus), "--profile", "local", "--json"]) == EXIT_OK
    assert json.loads(capsys.readouterr().out)["persist_dir"] == ".nanorag"
    main(["query", "q", "--profile", "local", "--embedding-model", "other-model"])
    seen = fake_generator.settings_seen[-1]
    assert (seen.generator_preset, seen.embedding_model) == ("ollama", "other-model")


# --- exit codes ---------------------------------------------------------------


def test_exit_codes_follow_the_error_hierarchy(
    corpus, fake_generator, monkeypatch, capsys
):
    _ingest(corpus)
    capsys.readouterr()

    def raising(exc):
        def pick(settings, **_):
            raise exc

        return pick

    cases = [
        (ConfigError("no generator available"), EXIT_CONFIG),
        (AuthError("rejected key", status=401), EXIT_PROVIDER),
        (QuotaExhausted("daily cap"), EXIT_PROVIDER),
        (RetrievalError("odd"), EXIT_FAILURE),
    ]
    for exc, expected in cases:
        monkeypatch.setattr(pipeline, "default_generator", raising(exc))
        assert _query("q") == expected, exc
        out, err = capsys.readouterr()
        assert out == ""
        assert err.startswith(f"nanorag: error: {exc.message}")


def test_provider_failure_during_generation_is_exit_4(corpus, fake_generator, capsys):
    _ingest(corpus)
    fake_generator._responses[:] = [AuthError("bad key", api_key="secret-value")]
    assert _query("q") == EXIT_PROVIDER
    err = capsys.readouterr().err
    assert "bad key" in err and "secret-value" not in err


def test_index_built_with_another_model_is_exit_5(corpus, fake_generator, capsys):
    _ingest(corpus)
    code = main(["query", "q", "--persist-dir", "idx", "--embedding-model", "other"])
    assert code == EXIT_STORE
    assert "different embedding model" in capsys.readouterr().err


def test_bad_profile_is_a_config_error(cwd, fake_generator, capsys):
    assert main(["inspect", "--profile", "nope"]) == EXIT_CONFIG
    assert "unknown profile" in capsys.readouterr().err


def test_unexpected_exceptions_propagate_as_bugs(corpus, fake_generator, monkeypatch):
    _ingest(corpus)

    def boom(settings, **_):
        raise RuntimeError("a bug")

    monkeypatch.setattr(pipeline, "default_generator", boom)
    with pytest.raises(RuntimeError, match="a bug"):
        _query("q")


# --- inspect ------------------------------------------------------------------


def test_inspect_without_an_index_is_exit_5_and_creates_nothing(
    cwd, fake_generator, capsys
):
    assert main(["inspect", "--persist-dir", "idx"]) == EXIT_STORE
    assert "no index at idx" in capsys.readouterr().err
    assert not (cwd / "idx").exists()


def test_inspect_reports_model_counts_and_documents(corpus, fake_generator, capsys):
    _ingest(corpus)
    capsys.readouterr()
    assert main(["inspect", "--persist-dir", "idx", "--json"]) == EXIT_OK
    payload = json.loads(capsys.readouterr().out)
    assert set(payload) == {"persist_dir", "index", "documents", "chunks", "sources"}
    assert payload["index"] == {"model_id": "fake-model", "dim": 64}
    assert payload["documents"] == 3 and payload["chunks"] == 3
    assert [d["source_uri"] for d in payload["sources"]] == sorted(
        d["source_uri"] for d in payload["sources"]
    )
    assert {d["source_uri"] for d in payload["sources"]} == {
        "cats.txt",
        "dogs.txt",
        "notes/fish.md",
    }
    for d in payload["sources"]:
        assert set(d) == {"doc_id", "source_uri", "content_hash", "chunks", "metadata"}
        assert d["chunks"] == 1

    assert main(["inspect", "--persist-dir", "idx"]) == EXIT_OK
    out = capsys.readouterr().out
    assert "embedding model: fake-model (64-dim)" in out
    assert "documents: 3, chunks: 3" in out
    assert "notes/fish.md" in out


def test_inspect_an_index_with_no_documents(cwd, fake_generator, capsys):
    (cwd / "empty").mkdir()
    _ingest(cwd / "empty")
    capsys.readouterr()
    assert main(["inspect", "--persist-dir", "idx", "--json"]) == EXIT_OK
    payload = json.loads(capsys.readouterr().out)
    assert payload["index"] is None  # no embeddings were ever written
    assert payload["documents"] == 0 and payload["sources"] == []
    assert main(["inspect", "--persist-dir", "idx"]) == EXIT_OK
    out = capsys.readouterr().out
    assert "embedding model: none yet" in out and "documents (" not in out


# --- .env, stdout encoding, entry points ---------------------------------------


def test_dotenv_exports_only_unset_variables_and_never_prints_them(
    cwd, fake_generator, monkeypatch, capsys
):
    monkeypatch.delenv("NANORAG_TEST_A", raising=False)
    monkeypatch.setenv("NANORAG_TEST_B", "from-the-real-environment")
    (cwd / ".env").write_text(
        "# a comment\n\nNANORAG_TEST_A = 'quoted-value'\n"
        'export NANORAG_TEST_B="from-the-file"\nnot a pair\n',
        encoding="utf-8",
    )
    main(["inspect", "--persist-dir", "idx"])
    import os

    assert os.environ["NANORAG_TEST_A"] == "quoted-value"
    assert os.environ["NANORAG_TEST_B"] == "from-the-real-environment"
    out, err = capsys.readouterr()
    assert "quoted-value" not in out + err
    monkeypatch.delenv("NANORAG_TEST_A")

    other = cwd / "keys.env"
    other.write_text("NANORAG_TEST_C=c\n", encoding="utf-8")
    monkeypatch.delenv("NANORAG_TEST_C", raising=False)
    main(["inspect", "--persist-dir", "idx", "--env-file", str(other)])
    assert os.environ["NANORAG_TEST_C"] == "c"
    monkeypatch.delenv("NANORAG_TEST_C")


def test_load_dotenv_counts_and_tolerates_a_missing_file(tmp_path, monkeypatch):
    assert load_dotenv(tmp_path / "absent") == 0
    monkeypatch.delenv("NANORAG_TEST_D", raising=False)
    (tmp_path / "e").write_text("NANORAG_TEST_D=1\nNANORAG_TEST_D=2\n", "utf-8")
    assert load_dotenv(tmp_path / "e") == 1  # the second line is "already set"
    monkeypatch.delenv("NANORAG_TEST_D")


def test_utf8_stdout_tolerates_a_stream_without_reconfigure(monkeypatch):
    monkeypatch.setattr(sys, "stdout", io.StringIO())
    _utf8_stdout()  # must not raise


def test_version_flag(capsys):
    with pytest.raises(SystemExit) as info:
        main(["--version"])
    assert info.value.code == 0
    from nanorag import __version__

    assert capsys.readouterr().out.strip() == f"nanorag {__version__}"


def test_python_dash_m_entry_point_runs():
    out = subprocess.run(
        [sys.executable, "-m", "nanorag.cli", "--version"],
        capture_output=True,
        text=True,
        check=True,
    )
    assert out.stdout.startswith("nanorag ")
