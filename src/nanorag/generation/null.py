"""``NullGenerator`` — the ``Generator`` for a pipeline that only ingests.

The write path needs an embedder and the stores, never a model; the read
path is the only thing that does. ``Rag`` still takes a generator, so a
process that builds an index and never queries it — ``nanorag ingest``,
an indexing job in CI — wires this null object in place of a provider and
never has to hold a key or reach a server for work that does not need
one. It refuses the first ``generate`` call with a ``ConfigError``, so
a query against it fails loudly rather than answering with nothing.

``context_window`` is a plausible size so that ``Rag.query``'s budget
arithmetic still runs (with ``1`` every context would be empty and the
pipeline would abstain *before* reaching this generator, hiding the
misconfiguration behind a valid-looking "I don't know").
"""

from __future__ import annotations

from nanorag.errors import ConfigError
from nanorag.generation.base import Generation
from nanorag.prompting.templates import Prompt


class NullGenerator:
    """A ``Generator`` that always raises ``ConfigError`` — see the module docstring."""

    provider = "none"
    model = "none"
    context_window = 8_192
    max_output_tokens = 1_024

    def generate(self, prompt: Prompt) -> Generation:
        """Raise ``ConfigError``: this pipeline was wired without a generator."""
        raise ConfigError(
            "no generator was configured for this pipeline; it can ingest but not "
            "answer — construct it with a real generator (or, on the command line, "
            "run `nanorag query` rather than `nanorag ingest`)"
        )

    def __repr__(self) -> str:
        """Return ``NullGenerator()``."""
        return "NullGenerator()"
