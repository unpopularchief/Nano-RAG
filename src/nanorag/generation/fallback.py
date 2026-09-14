"""``FallbackGenerator`` — try the next provider when the current one fails.

The free tiers this project runs on have daily caps, retire model names
without notice, and go down. plan.md §6 makes fallback a ``Generator``
responsibility and §18 F6 fixes the triggers: ``QuotaExhausted``, any
``ProviderError`` (including the 404 a retired model name produces), and —
since the chain exists precisely so a query still gets answered — a
``RateLimitError`` or ``TransientError`` that survived the client's own
retries. The one exception is ``AuthError``: a rejected key is a
configuration problem to surface, not to paper over by quietly answering
from a different account.

``Usage.provider`` on the returned generation always names the provider
that actually served the query, so a caller never has to guess which one
answered (plan.md §5). Every hand-off is logged at ``WARNING``.

The chain's ``context_window`` and ``max_output_tokens`` are the minimum
across its members: whichever member ends up serving, the prompt the
pipeline budgeted must fit.
"""

from __future__ import annotations

import logging

from nanorag.errors import AuthError, ProviderError
from nanorag.generation.base import Generation, Generator
from nanorag.prompting.templates import Prompt

log = logging.getLogger("nanorag.generation")


class FallbackGenerator:
    """A ``Generator`` that tries *generators* in order until one answers.

    Parameters
    ----------
    primary
        The first generator to try; its ``provider`` and ``model`` are the
        chain's nominal ones.
    *fallbacks
        Tried in order after *primary* fails with a fallback-worthy error.

    """

    def __init__(self, primary: Generator, *fallbacks: Generator) -> None:
        """Record the chain and take the tightest window across it."""
        self.generators: tuple[Generator, ...] = (primary, *fallbacks)
        self.provider = primary.provider
        self.model = primary.model
        self.context_window = min(g.context_window for g in self.generators)
        self.max_output_tokens = min(g.max_output_tokens for g in self.generators)

    def __repr__(self) -> str:
        """Return the chain's provider names in order."""
        names = " -> ".join(g.provider for g in self.generators)
        return f"FallbackGenerator({names})"

    def generate(self, prompt: Prompt) -> Generation:
        """Return the first member's generation that succeeds.

        Raises
        ------
        AuthError
            Immediately, from whichever member raised it — no fallback.
        ProviderError
            The last member's error, when every member failed.

        """
        last: ProviderError | None = None
        for index, generator in enumerate(self.generators):
            try:
                return generator.generate(prompt)
            except AuthError:
                raise
            except ProviderError as exc:
                last = exc
                if index + 1 < len(self.generators):
                    nxt = self.generators[index + 1]
                    log.warning(
                        "%s failed (%s: %s); falling back to %s",
                        generator.provider,
                        type(exc).__name__,
                        exc.message,
                        nxt.provider,
                    )
        assert last is not None  # the loop ran at least once
        raise last
