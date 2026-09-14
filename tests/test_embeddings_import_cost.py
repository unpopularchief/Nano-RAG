"""Default-suite tests: the [local] extra is optional, never required to import.

These do not need fastembed/onnxruntime installed and must pass even when
they are not — unlike ``tests/test_embeddings_local.py`` (``-m local``),
which exercises the real model and is skipped by default.
"""

import builtins
import subprocess
import sys

import pytest

from nanorag.embeddings.local import FastEmbedEmbedder
from nanorag.errors import ConfigError


def test_import_nanorag_embeddings_pulls_in_no_local_backend():
    code = (
        "import sys, nanorag.embeddings; "
        "heavy = {'fastembed', 'onnxruntime'}; "
        "print(sorted(heavy & set(sys.modules)))"
    )
    out = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True, check=True
    )
    assert out.stdout.strip() == "[]", out.stdout


def test_constructing_fastembed_embedder_without_the_extra_raises_config_error(
    monkeypatch,
):
    real_import = builtins.__import__

    def _no_fastembed(name, *args, **kwargs):
        if name == "fastembed" or name.startswith("fastembed."):
            raise ImportError("simulated: fastembed is not installed")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", _no_fastembed)

    with pytest.raises(ConfigError, match=r"nanorag\[local\]"):
        FastEmbedEmbedder()
