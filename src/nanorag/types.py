"""Core value types — frozen, slotted dataclasses validated on construction.

``Document``, ``Chunk`` and ``ScoredChunk`` are the stable spine of the
library. ``Answer``, ``Citation``, ``Usage`` and ``Timings`` are the query
path's output types, **frozen at Gate C / ``v0.1.0``** against the working
read path rather than one phase ahead of it (plan.md §18 F11): from here
on, a field change is a breaking change. ``IngestReport`` and ``LoadIssue``
are the write path's counterpart, introduced in Phase B; they stay
informally unstable, since deletion/update semantics (Phase E) will add
fields.

Metadata is a flat ``str -> JSON scalar`` mapping. Nesting is rejected: it
breaks the filter grammar's compilation to bound SQL parameters and buys
nothing (plan.md §11). On construction each metadata mapping is copied into a
read-only :class:`types.MappingProxyType`, so an instance cannot be mutated
through it after the fact.
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass, field, fields
from types import MappingProxyType
from typing import Any

#: The only value types permitted in a metadata mapping.
JsonScalar = str | int | float | bool | None


def _freeze_metadata(metadata: Mapping[str, JsonScalar]) -> Mapping[str, JsonScalar]:
    """Validate a metadata mapping and return a read-only copy.

    Raises
    ------
    TypeError
        A key is not a ``str``, or a value is not a JSON scalar.
    ValueError
        A ``float`` value is NaN or infinite (not representable in JSON).

    """
    frozen: dict[str, JsonScalar] = {}
    for key, value in metadata.items():
        if not isinstance(key, str):
            raise TypeError(
                f"metadata keys must be str, got {type(key).__name__}: {key!r}"
            )
        # bool is a subclass of int, so it passes the int check; None is handled
        # separately. Anything else (list, dict, bytes, ...) is rejected.
        if value is not None and not isinstance(value, str | int | float):
            raise TypeError(
                f"metadata[{key!r}] must be a JSON scalar "
                f"(str, int, float, bool, None), got {type(value).__name__}"
            )
        if isinstance(value, float) and not math.isfinite(value):
            raise ValueError(f"metadata[{key!r}] float must be finite, got {value!r}")
        frozen[key] = value
    return MappingProxyType(frozen)


@dataclass(frozen=True, slots=True)
class Document:
    """A source document: its text plus the metadata needed to track it.

    Attributes
    ----------
    doc_id
        Stable id, normally :func:`nanorag.hashing.stable_doc_id` of
        ``source_uri`` (or user-supplied).
    source_uri
        Canonical location of the source (a path or URL).
    text
        The document's full extracted text.
    content_hash
        :func:`nanorag.hashing.content_hash` of ``text`` — drives change
        detection on re-ingest.
    metadata
        Flat ``str -> JSON scalar`` map. Read-only after construction.

    """

    doc_id: str
    source_uri: str
    text: str
    content_hash: str
    metadata: Mapping[str, JsonScalar] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """Validate the field values (see the class docstring)."""
        for name in ("doc_id", "source_uri", "content_hash"):
            if not getattr(self, name):
                raise ValueError(f"Document.{name} must be a non-empty string")
        object.__setattr__(self, "metadata", _freeze_metadata(self.metadata))

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serialisable dict of this document."""
        return {
            "doc_id": self.doc_id,
            "source_uri": self.source_uri,
            "text": self.text,
            "content_hash": self.content_hash,
            "metadata": dict(self.metadata),
        }


@dataclass(frozen=True, slots=True)
class Chunk:
    """A contiguous span of a document, the unit of retrieval.

    Attributes
    ----------
    chunk_id
        :func:`nanorag.hashing.chunk_id` of ``(doc_id, ordinal, text)``.
    doc_id
        Id of the owning document.
    ordinal
        Zero-based position of this chunk within the document.
    text
        The chunk's text.
    start_char, end_char
        Half-open offsets into the owning ``Document.text``. Slicing
        ``document.text[start_char:end_char]`` reproduces this chunk for plain
        text; they are what let a citation point at an exact span. Produced by
        the chunkers (Phase B); consumed by citations (Phase D).
    token_count
        Token count under the chunker's counter.
    metadata
        Flat ``str -> JSON scalar`` map. Read-only after construction.

    """

    chunk_id: str
    doc_id: str
    ordinal: int
    text: str
    start_char: int
    end_char: int
    token_count: int
    metadata: Mapping[str, JsonScalar] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """Validate the field values (see the class docstring)."""
        if not self.chunk_id or not self.doc_id:
            raise ValueError("Chunk.chunk_id and Chunk.doc_id must be non-empty")
        if self.ordinal < 0:
            raise ValueError(f"Chunk.ordinal must be >= 0, got {self.ordinal}")
        if self.start_char < 0:
            raise ValueError(f"Chunk.start_char must be >= 0, got {self.start_char}")
        if self.end_char < self.start_char:
            raise ValueError(
                f"Chunk.end_char ({self.end_char}) must be >= "
                f"start_char ({self.start_char})"
            )
        if self.token_count < 0:
            raise ValueError(f"Chunk.token_count must be >= 0, got {self.token_count}")
        object.__setattr__(self, "metadata", _freeze_metadata(self.metadata))

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serialisable dict of this chunk."""
        return {
            "chunk_id": self.chunk_id,
            "doc_id": self.doc_id,
            "ordinal": self.ordinal,
            "text": self.text,
            "start_char": self.start_char,
            "end_char": self.end_char,
            "token_count": self.token_count,
            "metadata": dict(self.metadata),
        }


@dataclass(frozen=True, slots=True)
class ScoredChunk:
    """A chunk with a retrieval score and its provenance.

    Wraps rather than subclasses ``Chunk`` — there is one chunk type.

    Attributes
    ----------
    chunk
        The retrieved chunk.
    score
        The score under ``source``'s ranking (cosine, BM25, rerank). Higher is
        more relevant; the scale is not comparable across sources.
        Where the score came from: ``"dense"``, ``"bm25"``, ``"hybrid:rrf"``,
        ``"rerank:jina"``, ``"rerank:local"``, … — provenance for debugging
        a ranking.

    """

    chunk: Chunk
    score: float
    source: str

    def __post_init__(self) -> None:
        """Validate the field values (see the class docstring)."""
        if not math.isfinite(self.score):
            raise ValueError(f"ScoredChunk.score must be finite, got {self.score!r}")
        if not self.source:
            raise ValueError("ScoredChunk.source must be a non-empty string")

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serialisable dict of this scored chunk."""
        return {
            "chunk": self.chunk.to_dict(),
            "score": self.score,
            "source": self.source,
        }


@dataclass(frozen=True, slots=True)
class Citation:
    """A resolved ``[n]`` marker linking answer text back to a source span.

    Stable since ``v0.1.0``. Populated by
    :func:`nanorag.citations.parser.parse_citations`, wired into
    ``Rag.query`` in Phase D (session D1).

    Attributes
    ----------
    label
        The marker number as it appears in ``Answer.text`` (1-based).
    chunk_id, doc_id
        The chunk and document the marker resolved to.
    source_uri
        ``source_uri`` of that document — what a reader opens.
    start_char, end_char
        Half-open offsets into that document's text: the exact cited span.

    """

    label: int
    chunk_id: str
    doc_id: str
    source_uri: str
    start_char: int
    end_char: int

    def __post_init__(self) -> None:
        """Validate the field values (see the class docstring)."""
        if self.label < 1:
            raise ValueError(f"Citation.label must be >= 1, got {self.label}")
        for name in ("chunk_id", "doc_id", "source_uri"):
            if not getattr(self, name):
                raise ValueError(f"Citation.{name} must be a non-empty string")
        if self.start_char < 0 or self.end_char < self.start_char:
            raise ValueError(
                f"Citation span invalid: [{self.start_char}, {self.end_char})"
            )

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serialisable dict of this citation."""
        return {
            "label": self.label,
            "chunk_id": self.chunk_id,
            "doc_id": self.doc_id,
            "source_uri": self.source_uri,
            "start_char": self.start_char,
            "end_char": self.end_char,
        }


@dataclass(frozen=True, slots=True)
class Usage:
    """Token, cost and provider accounting for one query.

    Stable since ``v0.1.0``.

    Attributes
    ----------
    provider
        The provider that actually served the query, after any fallback — the
        machine-readable half of "which provider answered" (plan.md §5).
    model
        The model name that served it.
    prompt_tokens, completion_tokens, total_tokens
        Token counts, reconciled against the provider's returned ``usage``
        where available.
    cost_usd
        Estimated cost in USD; ``0.0`` for free tiers and local models.

    """

    provider: str
    model: str
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int
    cost_usd: float = 0.0

    def __post_init__(self) -> None:
        """Validate the field values (see the class docstring)."""
        if not self.provider:
            raise ValueError("Usage.provider must be a non-empty string")
        for name in ("prompt_tokens", "completion_tokens", "total_tokens"):
            if getattr(self, name) < 0:
                raise ValueError(f"Usage.{name} must be >= 0")
        if not math.isfinite(self.cost_usd) or self.cost_usd < 0:
            raise ValueError(f"Usage.cost_usd must be >= 0, got {self.cost_usd!r}")

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serialisable dict of this usage record."""
        return {
            "provider": self.provider,
            "model": self.model,
            "prompt_tokens": self.prompt_tokens,
            "completion_tokens": self.completion_tokens,
            "total_tokens": self.total_tokens,
            "cost_usd": self.cost_usd,
        }


@dataclass(frozen=True, slots=True)
class Timings:
    """Wall-clock milliseconds spent in each pipeline stage.

    Stable since ``v0.1.0``. All fields default to ``0.0`` so a partial
    pipeline is easy to record; a stage the pipeline does not run yet
    (``rerank_ms`` until Phase F) or does not separate out (``embed_ms`` —
    the query embedding is inside ``retrieve_ms``) reads ``0.0``.
    """

    embed_ms: float = 0.0
    retrieve_ms: float = 0.0
    rerank_ms: float = 0.0
    context_ms: float = 0.0
    generate_ms: float = 0.0
    total_ms: float = 0.0

    def __post_init__(self) -> None:
        """Validate the field values (see the class docstring)."""
        for f in fields(self):
            value = getattr(self, f.name)
            if not math.isfinite(value) or value < 0:
                raise ValueError(f"Timings.{f.name} must be >= 0, got {value!r}")

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serialisable dict of these timings."""
        return {f.name: getattr(self, f.name) for f in fields(self)}


@dataclass(frozen=True, slots=True)
class Answer:
    """The result of a query: text, the chunks behind it, and accounting.

    **Stable since ``v0.1.0``** (frozen at Gate C, plan.md §18 F11).

    Attributes
    ----------
    text
        The generated answer.
    citations
        Resolved ``[n]`` markers, in label order.
    contexts
        Exactly the scored chunks that were placed in the prompt.
    insufficient_context
        Retrieval fell below the count/score floor
        (``nanorag.pipeline.insufficient_context``). With no usable context
        the pipeline never called the model and ``text`` is the fixed
        abstention; with thin context the model was still asked, under a
        prompt that tells it to abstain. Either way a caller never has to
        parse ``text`` to learn this.
    truncated
        One or more context blocks were dropped to fit the token budget.
    usage
        Token, cost and served-provider accounting.
    timings
        Per-stage wall-clock timing.

    """

    text: str
    citations: tuple[Citation, ...]
    contexts: tuple[ScoredChunk, ...]
    insufficient_context: bool
    truncated: bool
    usage: Usage
    timings: Timings

    def __post_init__(self) -> None:
        """Validate the field values (see the class docstring)."""
        if not isinstance(self.citations, tuple):
            raise TypeError("Answer.citations must be a tuple")
        if not isinstance(self.contexts, tuple):
            raise TypeError("Answer.contexts must be a tuple")

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serialisable dict of this answer."""
        return {
            "text": self.text,
            "citations": [c.to_dict() for c in self.citations],
            "contexts": [sc.to_dict() for sc in self.contexts],
            "insufficient_context": self.insufficient_context,
            "truncated": self.truncated,
            "usage": self.usage.to_dict(),
            "timings": self.timings.to_dict(),
        }


@dataclass(frozen=True, slots=True)
class LoadIssue:
    """A source a loader encountered but did not turn into a ``Document``.

    Used for both ``IngestReport.skipped`` (deliberately not attempted, e.g.
    an unrecognised extension) and ``IngestReport.failed`` (attempted and
    raised ``LoaderError``) — the two are told apart by which tuple they are
    in, not by a field on this type.

    Attributes
    ----------
    source_uri
        The source that was not loaded.
    reason
        Human-readable explanation (not machine-parsed).

    """

    source_uri: str
    reason: str

    def __post_init__(self) -> None:
        """Validate the field values (see the class docstring)."""
        if not self.source_uri:
            raise ValueError("LoadIssue.source_uri must be a non-empty string")
        if not self.reason:
            raise ValueError("LoadIssue.reason must be a non-empty string")

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serialisable dict of this issue."""
        return {"source_uri": self.source_uri, "reason": self.reason}


@dataclass(frozen=True, slots=True)
class IngestReport:
    """The outcome of loading a batch of sources (a directory walk, so far).

    Attributes
    ----------
    loaded
        Documents successfully produced.
    skipped
        Sources deliberately not attempted (e.g. an unrecognised extension,
        a symlink, an excluded path) — never passed to a ``Loader``.
    failed
        Sources a ``Loader`` attempted and raised ``LoaderError`` on. One
        failure never aborts the batch — every other source is still
        attempted.

    """

    loaded: tuple[Document, ...]
    skipped: tuple[LoadIssue, ...]
    failed: tuple[LoadIssue, ...]

    def __post_init__(self) -> None:
        """Validate the field values (see the class docstring)."""
        for name in ("loaded", "skipped", "failed"):
            if not isinstance(getattr(self, name), tuple):
                raise TypeError(f"IngestReport.{name} must be a tuple")

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serialisable dict of this report."""
        return {
            "loaded": [d.to_dict() for d in self.loaded],
            "skipped": [i.to_dict() for i in self.skipped],
            "failed": [i.to_dict() for i in self.failed],
        }


@dataclass(frozen=True, slots=True)
class SyncReport:
    """The outcome of reconciling the store against one directory tree.

    Produced by ``Rag.sync_path`` (Phase E, session E2): a dry run (the
    default) returns this as a plan with nothing written; ``apply=True``
    executes it first.

    Attributes
    ----------
    added, updated, unchanged, deleted
        Source URIs in each category, from comparing the store's current
        documents against a fresh walk of the root: ``added`` is on disk
        but not yet stored; ``updated`` is in both with a changed
        ``content_hash``; ``unchanged`` is in both with the same hash (a
        no-op either way); ``deleted`` is stored but no longer produced
        by the walk.
    skipped, failed
        As :class:`IngestReport` — sources the walk did not attempt or
        could not load. Never affects the four lists above.
    applied
        ``False`` on a dry run: nothing was written, the lists above are
        the plan. ``True``: the store now matches the walk (unless
        ``over_delete_guard`` blocked it — see below).
    over_delete_guard
        ``True`` when ``deleted`` exceeds the sync's ``max_delete_fraction``
        of the store's document count from *before* this sync — the
        condition under which ``apply=True`` refuses to run (plan.md §18
        F9). Computed on every sync, dry run or not, so a caller can see a
        would-be-blocked deletion without ever passing ``apply=True``.

    """

    added: tuple[str, ...]
    updated: tuple[str, ...]
    unchanged: tuple[str, ...]
    deleted: tuple[str, ...]
    skipped: tuple[LoadIssue, ...]
    failed: tuple[LoadIssue, ...]
    applied: bool
    over_delete_guard: bool

    def __post_init__(self) -> None:
        """Validate the field values (see the class docstring)."""
        for name in ("added", "updated", "unchanged", "deleted", "skipped", "failed"):
            if not isinstance(getattr(self, name), tuple):
                raise TypeError(f"SyncReport.{name} must be a tuple")

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serialisable dict of this report."""
        return {
            "added": list(self.added),
            "updated": list(self.updated),
            "unchanged": list(self.unchanged),
            "deleted": list(self.deleted),
            "skipped": [i.to_dict() for i in self.skipped],
            "failed": [i.to_dict() for i in self.failed],
            "applied": self.applied,
            "over_delete_guard": self.over_delete_guard,
        }
