"""The examples stay runnable: ``offline_fakes.py`` runs end to end in a
subprocess with no model, key or network (the "fakes in CI" leg of the
Phase C acceptance), and both examples stay within plan.md §14's 40-line
budget. ``quickstart.py`` needs a real embedder and generator, so here it
is only compiled and checked for the README's five lines."""

import ast
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
EXAMPLES = ROOT / "examples"


def test_offline_fakes_example_runs_without_network():
    # Sockets are blocked *inside* the child — patching this process would
    # prove nothing about a separate interpreter.
    code = "\n".join(
        [
            "import runpy, sys; from tests.conftest import blocked_sockets",
            "with blocked_sockets():",
            "    runpy.run_path(sys.argv[1], run_name='__main__')",
        ]
    )
    out = subprocess.run(
        [sys.executable, "-c", code, str(EXAMPLES / "offline_fakes.py")],
        capture_output=True,
        text=True,
        cwd=ROOT,
        check=True,
        timeout=120,
    )
    doc_names = [p.name for p in (ROOT / "docs").glob("**/*.md")]
    n_docs = len(doc_names)
    assert f"ingested {n_docs} documents" in out.stdout
    assert "A: Keys come from environment variables only. [1]" in out.stdout
    label_line = next(
        line for line in out.stdout.splitlines() if line.strip().startswith("[1]")
    )
    assert any(name in label_line for name in doc_names), label_line
    assert "=== BEGIN UNTRUSTED CONTEXT " in out.stdout


def test_examples_stay_within_forty_lines():
    for path in EXAMPLES.glob("*.py"):
        assert len(path.read_text(encoding="utf-8").splitlines()) <= 40, path.name


def test_quickstart_compiles_and_matches_the_readme_flow():
    source = (EXAMPLES / "quickstart.py").read_text(encoding="utf-8")
    ast.parse(source)  # syntax only; it needs a model and a key to run
    for line in (
        "from nanorag import Rag",
        'Rag.from_defaults(persist_dir=".nanorag")',
        'rag.ingest_path("docs/", glob="**/*.md")',
        "rag.query(",
        "answer.text",
    ):
        assert line in source, line
    readme = (ROOT / "README.md").read_text(encoding="utf-8")
    assert 'rag = Rag.from_defaults(persist_dir=".nanorag")' in readme
    assert 'rag.ingest_path("docs/", glob="**/*.md")' in readme
