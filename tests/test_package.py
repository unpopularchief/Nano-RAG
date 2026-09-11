"""Package-level invariants: version, public surface, import cost."""

import importlib
import subprocess
import sys

import nanorag


def test_version_is_exposed():
    assert nanorag.__version__ == "0.0.1"


def test_public_names_are_importable_from_the_top_level():
    for name in nanorag.__all__:
        assert hasattr(nanorag, name), name


def test_errors_and_types_are_reexported():
    from nanorag import Answer, Document, NanoRagError, RateLimitError

    assert issubclass(RateLimitError, NanoRagError)
    assert Document("d", "s", "t", "h").doc_id == "d"
    assert Answer.__name__ == "Answer"


def test_import_nanorag_pulls_in_no_heavy_or_provider_module():
    # Gate A invariant: a fresh `import nanorag` drags in no HTTP client,
    # embedder runtime or tokenizer. Checked in a clean interpreter so other
    # tests' imports do not mask a regression.
    code = (
        "import sys, nanorag; "
        "heavy = {'httpx', 'fastembed', 'onnxruntime', 'tiktoken', 'numpy'}; "
        "print(sorted(heavy & set(sys.modules)))"
    )
    out = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True, check=True
    )
    assert out.stdout.strip() == "[]", out.stdout


def test_every_submodule_imports_cleanly():
    for mod in (
        "nanorag.types",
        "nanorag.errors",
        "nanorag.hashing",
        "nanorag.tokens",
        "nanorag.config",
    ):
        importlib.import_module(mod)
