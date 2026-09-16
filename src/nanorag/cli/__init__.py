"""The ``nanorag`` command line (Phase D, session D2).

``nanorag ingest | query | inspect``, each with ``--json`` for a stable,
documented payload (``docs/cli.md``) and exit codes that follow the error
hierarchy. Installed as the ``nanorag`` console script; also runnable as
``python -m nanorag.cli``. See :mod:`nanorag.cli.app` for the contract.
"""

from __future__ import annotations

from nanorag.cli.app import main

__all__ = ["main"]
