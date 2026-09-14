"""Nonce fences — how untrusted retrieved text is delimited inside a prompt.

Retrieved text is **data**, not instructions (plan.md §11 "Prompt
injection"). A document can contain anything, including a line that looks
like the end of the context and the start of a new "system" instruction. A
fixed delimiter cannot stop that, because an attacker who has read this file
knows the delimiter. A **per-request nonce** in both fence lines can: the
system prompt names the nonce, so only markers carrying it are real, and the
text inside the fence cannot forge one it has never seen.

Two properties are enforced, not hoped for:

- :func:`make_nonce` never returns a value that already occurs in the text
  it is asked to avoid (a ``secrets``-random 128-bit value collides with
  probability ~0, but the check costs one ``in`` and turns "practically
  impossible" into "impossible").
- :func:`fence` refuses a body that contains its nonce, so a fence line can
  never be forged from inside the fence.

This is risk reduction, not prevention — a model may still be talked into
misbehaving by content it was told to treat as data. Phase H adds a
heuristic scanner; nothing here claims immunity.
"""

from __future__ import annotations

import secrets

#: Bytes of randomness per nonce; rendered as twice as many hex characters.
NONCE_BYTES = 16

#: The fence line templates. Both name the nonce so a forged fence would
#: need it, and both say "untrusted" so the marker is self-describing in a
#: prompt dump.
BEGIN_FENCE = "=== BEGIN UNTRUSTED CONTEXT {nonce} ==="
END_FENCE = "=== END UNTRUSTED CONTEXT {nonce} ==="


def make_nonce(*, avoid: str = "") -> str:
    """Return a fresh random nonce that does not occur in *avoid*.

    Parameters
    ----------
    avoid
        Text the nonce must not be a substring of — pass everything that
        will sit inside or around the fence (the context and the question)
        so the closing fence is unforgeable by construction.

    Returns
    -------
    str
        ``2 * NONCE_BYTES`` lowercase hex characters.

    """
    while True:
        nonce = secrets.token_hex(NONCE_BYTES)
        if nonce not in avoid:
            return nonce


def fence(body: str, nonce: str) -> str:
    """Wrap *body* between the begin/end fence lines carrying *nonce*.

    Parameters
    ----------
    body
        The untrusted text. Placed verbatim — never escaped or altered, so
        the prompt carries exactly the chunk bytes the context recorded.
    nonce
        A value from :func:`make_nonce`.

    Returns
    -------
    str
        ``BEGIN`` line, newline, *body*, newline, ``END`` line. An empty
        *body* still yields both fence lines with an empty line between.

    Raises
    ------
    ValueError
        *nonce* is empty, or *body* contains *nonce* — the caller did not
        derive the nonce with ``make_nonce(avoid=body)``.

    """
    if not nonce:
        raise ValueError("nonce must be non-empty")
    if nonce in body:
        raise ValueError("fenced body must not contain its own nonce")
    return "\n".join(
        (BEGIN_FENCE.format(nonce=nonce), body, END_FENCE.format(nonce=nonce))
    )
