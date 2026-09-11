"""Shared test configuration.

Points ``tiktoken`` at the vendored encoding cache committed under
``tests/data/tiktoken_cache/`` so the default suite never reaches the network
(plan.md §16 A1, Gate A checklist). Set before any test imports ``tiktoken``.
"""

import os
from pathlib import Path

_TIKTOKEN_CACHE = Path(__file__).parent / "data" / "tiktoken_cache"
os.environ.setdefault("TIKTOKEN_CACHE_DIR", str(_TIKTOKEN_CACHE))
