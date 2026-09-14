"""Prompting: system prompt + nonce-fenced untrusted context + question.

``fencing.py`` owns the per-request nonce and the fence lines;
``templates.py`` owns the system-prompt instruction hierarchy and assembles
the ``Prompt``. ``injection.py`` (the heuristic scanner) is Phase H. Pure
Python — no provider, no ``numpy``.
"""

from nanorag.prompting.fencing import BEGIN_FENCE, END_FENCE, fence, make_nonce
from nanorag.prompting.templates import (
    DEFAULT_SYSTEM_PROMPT,
    INSUFFICIENT_CONTEXT_TEXT,
    Prompt,
    PromptBuilder,
)

__all__ = [
    "BEGIN_FENCE",
    "END_FENCE",
    "fence",
    "make_nonce",
    "DEFAULT_SYSTEM_PROMPT",
    "INSUFFICIENT_CONTEXT_TEXT",
    "Prompt",
    "PromptBuilder",
]
