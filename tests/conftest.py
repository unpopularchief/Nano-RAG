"""Shared test configuration.

Points ``tiktoken`` at the vendored encoding cache committed under
``tests/data/tiktoken_cache/`` so the default suite never reaches the network
(plan.md §16 A1, Gate A checklist). Set before any test imports ``tiktoken``.
"""

import os
import socket
from pathlib import Path

_TIKTOKEN_CACHE = Path(__file__).parent / "data" / "tiktoken_cache"
os.environ.setdefault("TIKTOKEN_CACHE_DIR", str(_TIKTOKEN_CACHE))


class blocked_sockets:
    """Context manager: any attempt to open a network connection raises.

    Patches ``socket.socket.connect``/``connect_ex`` rather than replacing
    ``socket.socket`` itself, since libraries construct real socket objects
    (for ``family``/``type``, ``isinstance`` checks, etc.) even on a call
    that never reaches the network — only the connection step should fail.
    Used to prove an "offline" claim (plan.md §9 Phase B acceptance: "no
    network access at all, verified by blocking sockets in the test") rather
    than merely asserting no networking library was imported.
    """

    def __enter__(self) -> "blocked_sockets":
        """Patch ``connect``/``connect_ex`` to raise; return self."""
        self._connect = socket.socket.connect
        self._connect_ex = socket.socket.connect_ex

        def _raise(*args: object, **kwargs: object) -> None:
            raise OSError("network access blocked in test")

        socket.socket.connect = _raise  # type: ignore[method-assign]
        socket.socket.connect_ex = _raise  # type: ignore[method-assign]
        return self

    def __exit__(self, *exc_info: object) -> None:
        """Restore the original ``connect``/``connect_ex``."""
        socket.socket.connect = self._connect  # type: ignore[method-assign]
        socket.socket.connect_ex = self._connect_ex  # type: ignore[method-assign]
