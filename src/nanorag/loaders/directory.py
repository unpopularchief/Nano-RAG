"""``DirectoryLoader`` — walk a directory tree into an ``IngestReport``.

Dispatches each file to a per-extension ``Loader`` (:mod:`nanorag.loaders.text`,
:mod:`nanorag.loaders.markdown`), never crosses a symlink, and matches
``glob``/``ignore`` patterns against POSIX-style paths *relative to the ingest
root* so that a fixed corpus produces the same ``source_uri`` (and therefore
the same ``doc_id``) on Windows and Linux alike.
"""

from __future__ import annotations

import os
from collections.abc import Mapping, Sequence
from pathlib import Path

from nanorag.errors import LoaderError
from nanorag.loaders.base import Loader
from nanorag.loaders.globbing import glob_match
from nanorag.loaders.markdown import MarkdownLoader
from nanorag.loaders.text import TextLoader
from nanorag.types import IngestReport, LoadIssue

#: Extension -> Loader used when the caller does not supply its own mapping.
DEFAULT_LOADERS: dict[str, Loader] = {
    ".txt": TextLoader(),
    ".md": MarkdownLoader(),
    ".markdown": MarkdownLoader(),
}


class DirectoryLoader:
    """Walks a directory tree, dispatching each file by extension.

    Guards against symlink loops and reading outside the ingest root by
    never descending into, or reading, a symlink (plan.md §13) — every
    symlinked file or directory is recorded in ``IngestReport.skipped``.

    Parameters
    ----------
    loaders
        Extension (lowercased, with leading dot) -> ``Loader``. Defaults to
        :data:`DEFAULT_LOADERS`.

    """

    def __init__(self, loaders: Mapping[str, Loader] | None = None) -> None:
        """Store *loaders* (see the class docstring)."""
        self.loaders: Mapping[str, Loader] = (
            loaders if loaders is not None else DEFAULT_LOADERS
        )

    def load_path(
        self,
        root: str | Path,
        *,
        glob: str = "**/*",
        ignore: Sequence[str] = (),
    ) -> IngestReport:
        """Walk *root* and load every matching file.

        Parameters
        ----------
        root
            Directory to walk.
        glob
            Pattern matched against each file's path relative to *root*
            (POSIX separators). Gitignore-style semantics: ``*`` matches
            within a single path segment (never crosses ``/``); ``**``
            (optionally followed by ``/``) matches zero or more segments.
            ``"**/*.md"`` matches both ``"a.md"`` and ``"sub/a.md"``;
            ``"*.md"`` matches only top-level ``"a.md"``. See
            :func:`nanorag.loaders.globbing.glob_match`.
        ignore
            Patterns (same semantics as *glob*) for paths to exclude
            entirely; an ignored file never appears in the returned report.

        Returns
        -------
        IngestReport
            ``loaded`` holds every successfully parsed ``Document``;
            ``skipped`` holds files never attempted (unrecognised extension,
            symlink, ignored); ``failed`` holds files a ``Loader`` raised
            ``LoaderError`` on. One failure never stops the walk.

        Raises
        ------
        LoaderError
            *root* does not exist or is not a directory.

        """
        root = Path(root)
        if not root.is_dir():
            raise LoaderError(f"not a directory: {root}", path=str(root))

        loaded = []
        skipped = []
        failed = []

        for dirpath, dirnames, filenames in os.walk(root, followlinks=False):
            current = Path(dirpath)
            # Prune symlinked and ignored subdirectories before descending.
            kept_dirnames = []
            for name in dirnames:
                dir_rel = (current / name).relative_to(root).as_posix()
                if (current / name).is_symlink():
                    skipped.append(LoadIssue(dir_rel + "/", "symlink"))
                    continue
                if any(
                    glob_match(dir_rel, pat) or glob_match(dir_rel + "/", pat)
                    for pat in ignore
                ):
                    continue
                kept_dirnames.append(name)
            dirnames[:] = kept_dirnames

            for name in filenames:
                file_path = current / name
                rel = file_path.relative_to(root).as_posix()

                if file_path.is_symlink():
                    skipped.append(LoadIssue(rel, "symlink"))
                    continue
                if any(glob_match(rel, pat) for pat in ignore):
                    continue
                if not glob_match(rel, glob):
                    continue

                loader = self.loaders.get(file_path.suffix.lower())
                if loader is None:
                    skipped.append(LoadIssue(rel, "unrecognized extension"))
                    continue

                try:
                    loaded.append(loader.load(file_path, source_uri=rel))
                except LoaderError as exc:
                    failed.append(LoadIssue(rel, str(exc)))

        return IngestReport(
            loaded=tuple(loaded), skipped=tuple(skipped), failed=tuple(failed)
        )
