"""``cli/sync.py``: the sync subcommand end to end with a stubbed embedder —
dry run vs. ``--apply``, the ``--json`` payload schema, offline (no
generator), and the delete-guard's exit code."""

import json

import pytest

from nanorag import pipeline
from nanorag.cli import main
from nanorag.cli.app import EXIT_OK, EXIT_STORE, EXIT_USAGE
from nanorag.embeddings import local
from tests.fakes import FakeEmbedder


class StubLocal(FakeEmbedder):
    """Stands in for ``FastEmbedEmbedder`` so ``from_defaults`` needs no model."""

    def __init__(self, model_id):
        super().__init__(dim=64, model_id=model_id)


@pytest.fixture
def cwd(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    for var in ("NANORAG_PROFILE", "NANORAG_PERSIST_DIR", "NANORAG_GENERATOR_PRESET"):
        monkeypatch.delenv(var, raising=False)
    monkeypatch.setattr(local, "FastEmbedEmbedder", StubLocal)
    return tmp_path


@pytest.fixture
def corpus(cwd):
    root = cwd / "corpus"
    root.mkdir()
    (root / "a.txt").write_text("alpha beta gamma", encoding="utf-8")
    (root / "b.txt").write_text("delta epsilon zeta", encoding="utf-8")
    return root


MODEL = ["--embedding-model", "fake-model"]


def _sync(corpus, *extra):
    return main(["sync", str(corpus), "--persist-dir", "idx", *MODEL, *extra])


def test_sync_is_offline_and_never_needs_a_generator(corpus, monkeypatch, capsys):
    def refuse(*_, **__):
        raise AssertionError("sync must not pick a generator")

    monkeypatch.setattr(pipeline, "default_generator", refuse)
    assert _sync(corpus) == EXIT_OK
    out, err = capsys.readouterr()
    assert "2 added, 0 updated, 0 deleted, 0 unchanged" in out
    assert "(dry run)" in out
    assert err == ""
    # A dry run writes nothing: the next dry run reports the same plan again.
    assert _sync(corpus) == EXIT_OK
    assert "2 added, 0 updated, 0 deleted, 0 unchanged" in capsys.readouterr().out


def test_sync_apply_writes_the_index(corpus, capsys):
    assert _sync(corpus, "--apply") == EXIT_OK
    out = capsys.readouterr().out
    assert "(applied)" in out
    assert (corpus.parent / "idx" / "nanorag.sqlite").is_file()

    assert _sync(corpus) == EXIT_OK  # nothing changed on disk
    out = capsys.readouterr().out
    assert "0 added, 0 updated, 0 deleted, 2 unchanged" in out

    (corpus / "a.txt").write_text("changed", encoding="utf-8")
    assert _sync(corpus) == EXIT_OK
    assert "  ~ a.txt" in capsys.readouterr().out


def test_sync_json_payload_reports_add_update_delete(corpus, capsys):
    assert _sync(corpus, "--apply", "--json") == EXIT_OK
    capsys.readouterr()

    (corpus / "a.txt").write_text("changed", encoding="utf-8")
    (corpus / "b.txt").unlink()
    (corpus / "c.txt").write_text("new file", encoding="utf-8")

    assert _sync(corpus, "--json") == EXIT_OK
    payload = json.loads(capsys.readouterr().out)
    assert set(payload) == {
        "added",
        "updated",
        "unchanged",
        "deleted",
        "skipped",
        "failed",
        "applied",
        "over_delete_guard",
        "persist_dir",
        "root",
        "glob",
        "ignore",
    }
    assert payload["added"] == ["c.txt"]
    assert payload["updated"] == ["a.txt"]
    assert payload["deleted"] == ["b.txt"]
    assert payload["applied"] is False
    assert payload["over_delete_guard"] is False


def test_sync_apply_refuses_past_max_delete_fraction(corpus, capsys):
    assert _sync(corpus, "--apply") == EXIT_OK
    capsys.readouterr()
    (corpus / "a.txt").unlink()
    (corpus / "b.txt").unlink()

    assert _sync(corpus, "--apply", "--max-delete-fraction", "0") == EXIT_STORE
    err = capsys.readouterr().err
    assert "max_delete_fraction" in err

    assert _sync(corpus, "--apply", "--max-delete-fraction", "1") == EXIT_OK
    out = capsys.readouterr().out
    assert "2 deleted" in out


def test_sync_dry_run_warns_on_stderr_when_over_the_delete_guard(corpus, capsys):
    assert _sync(corpus, "--apply") == EXIT_OK
    capsys.readouterr()
    (corpus / "a.txt").unlink()
    (corpus / "b.txt").unlink()

    assert _sync(corpus, "--max-delete-fraction", "0") == EXIT_OK
    err = capsys.readouterr().err
    assert "over max_delete_fraction" in err


def test_sync_root_that_is_not_a_directory_is_a_usage_error(cwd, capsys):
    with pytest.raises(SystemExit) as info:
        _sync(cwd / "missing")
    assert info.value.code == EXIT_USAGE
    assert "not a directory" in capsys.readouterr().err
