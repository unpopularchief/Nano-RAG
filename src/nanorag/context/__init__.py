"""Context building: ranked chunks -> a token-budgeted, ``[n]``-labelled context.

One module so far, ``builder.py``; ``dedup.py`` / ``ordering.py`` from the
eventual layout (plan.md §8) split out only if a second strategy for either
ever exists. Pure Python over the core types — no ``numpy``, no provider.
"""

from nanorag.context.builder import (
    BLOCK_SEPARATOR,
    Context,
    ContextBlock,
    ContextBuilder,
    render_block,
    render_context,
)

__all__ = [
    "BLOCK_SEPARATOR",
    "Context",
    "ContextBlock",
    "ContextBuilder",
    "render_block",
    "render_context",
]
