"""Package-level invariants: version, public surface, import cost."""

import importlib
import subprocess
import sys

import pytest

import nanorag


def test_version_is_exposed():
    assert nanorag.__version__ == "0.4.0"


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
        "nanorag.ratelimit",
        "nanorag.pipeline",
        "nanorag.observability",
        "nanorag.generation",
        "nanorag.cli",
    ):
        importlib.import_module(mod)


def test_rag_is_resolved_lazily_from_the_top_level():
    from nanorag import Rag
    from nanorag.pipeline import Rag as Direct

    assert Rag is Direct
    assert "Rag" in nanorag.__all__
    with pytest.raises(AttributeError):
        nanorag.no_such_name  # noqa: B018


def test_pipeline_and_generation_pull_no_provider_client_or_model_runtime():
    # Phase C invariant (plan.md §7): `from nanorag import Rag` may pull numpy
    # (the stores) but never httpx, fastembed, onnxruntime or tiktoken —
    # the concrete provider is imported by `from_defaults()` at call time.
    code = (
        "import sys; from nanorag import Rag; import nanorag.generation; "
        "heavy = {'httpx', 'fastembed', 'onnxruntime', 'tiktoken'}; "
        "print(sorted(heavy & set(sys.modules)))"
    )
    out = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True, check=True
    )
    assert out.stdout.strip() == "[]", out.stdout


def test_generation_clients_are_resolved_lazily():
    import nanorag.generation as generation

    assert generation.OpenAICompatGenerator.__name__ == "OpenAICompatGenerator"
    assert generation.GeminiGenerator.__name__ == "GeminiGenerator"
    with pytest.raises(AttributeError):
        generation.NoSuchGenerator  # noqa: B018


def test_rerankers_are_resolved_lazily():
    import nanorag.rerank as rerank

    assert rerank.JinaReranker.__name__ == "JinaReranker"
    assert rerank.LocalCrossEncoderReranker.__name__ == "LocalCrossEncoderReranker"
    with pytest.raises(AttributeError):
        rerank.NoSuchReranker  # noqa: B018


def test_pipeline_pulls_no_reranker_provider_or_model_runtime():
    # plan.md §9 Phase F: the default IdentityReranker has no dependencies,
    # so importing the pipeline must not pull httpx or fastembed just
    # because rerank/base.py and rerank/identity.py are imported eagerly.
    code = (
        "import sys; from nanorag import Rag; "
        "heavy = {'httpx', 'fastembed', 'onnxruntime'}; "
        "print(sorted(heavy & set(sys.modules)))"
    )
    out = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True, check=True
    )
    assert out.stdout.strip() == "[]", out.stdout
