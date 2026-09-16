"""Eval datasets: a frozen corpus plus graded questions, gold given as quotes.

Layout on disk::

    datasets/<name>/
    ├── corpus/       the documents, frozen — never edited after the
    │                 baseline is taken (a changed corpus is a new dataset)
    ├── dev.jsonl     one item per line (schema below)
    └── thresholds.json   committed baseline numbers (see ``report.py``)

Item schema (``dev.jsonl``)::

    {"id": "readme-003",
     "question": "Which embedding model does from_defaults() use?",
     "gold": [{"source_uri": "README.md", "quote": "Local ONNX embeddings"}],
     "answer": "bge-base-en-v1.5 via fastembed",          # optional
     "tags": ["readme", "embeddings"]}                       # optional

``gold`` names *where the answer is*: a document (``source_uri``, relative
to ``corpus/`` with POSIX separators — what ``DirectoryLoader`` records)
and a **quote** — a verbatim excerpt of that document, matched with
whitespace collapsed so it may span a line wrap. Each quote must occur
exactly once in its document; at load time it is resolved to character
offsets (:class:`ResolvedSpan`) and a quote that is missing or ambiguous
fails loudly. Gold is therefore defined on the document text, not on any
chunker's output, which is what lets one dataset grade a chunk-size sweep.
An item with ``"gold": []`` is **unanswerable** from the corpus; it grades
abstention rather than ranking.

The corpus is read through the library's own ``DirectoryLoader`` and then
line-ending- and Unicode-normalised (``hashing.normalize_text``): the same
bytes checked out with CRLF on Windows and LF on Linux must produce the
same chunks, or the committed baseline would not be reproducible across
the two.
"""

from __future__ import annotations

import json
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from nanorag.errors import EvaluationError
from nanorag.hashing import normalize_text
from nanorag.loaders.directory import DirectoryLoader
from nanorag.types import Document


@dataclass(frozen=True, slots=True)
class GoldSpan:
    """Where one answer lives: a document and a verbatim quote from it."""

    source_uri: str
    quote: str

    def __post_init__(self) -> None:
        """Validate the field values (see the class docstring)."""
        if not self.source_uri:
            raise ValueError("GoldSpan.source_uri must be a non-empty string")
        if not self.quote.strip():
            raise ValueError("GoldSpan.quote must be non-blank")


@dataclass(frozen=True, slots=True)
class EvalItem:
    """One graded question.

    Attributes
    ----------
    id
        Unique within the dataset.
    question
        The query text as a user would type it.
    gold
        Where the answer is; empty means unanswerable from this corpus.
    answer
        An optional short reference answer for lexical answer metrics.
    tags
        Free-form labels for slicing a report.

    """

    id: str
    question: str
    gold: tuple[GoldSpan, ...] = ()
    answer: str | None = None
    tags: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        """Validate the field values (see the class docstring)."""
        if not self.id:
            raise ValueError("EvalItem.id must be a non-empty string")
        if not self.question.strip():
            raise ValueError(f"EvalItem {self.id!r}: question must be non-blank")

    @property
    def answerable(self) -> bool:
        """True when the corpus holds at least one gold span for this item."""
        return bool(self.gold)


@dataclass(frozen=True, slots=True)
class ResolvedSpan:
    """A :class:`GoldSpan` located in its document: half-open char offsets."""

    doc_id: str
    source_uri: str
    start_char: int
    end_char: int

    def __post_init__(self) -> None:
        """Validate the field values (see the class docstring)."""
        if self.start_char < 0 or self.end_char <= self.start_char:
            raise ValueError(
                f"ResolvedSpan offsets invalid: [{self.start_char}, {self.end_char})"
            )

    def as_triple(self) -> tuple[str, int, int]:
        """Return ``(doc_id, start_char, end_char)`` for ``coverage_of``."""
        return (self.doc_id, self.start_char, self.end_char)


@dataclass(frozen=True, slots=True)
class EvalDataset:
    """A loaded dataset: items, the frozen corpus, and every gold span resolved.

    Attributes
    ----------
    name
        The dataset directory's name.
    items
        The graded questions, in file order.
    documents
        The corpus, normalised (see the module docstring).
    spans
        Item id → its resolved gold spans, in ``gold`` order.

    """

    name: str
    items: tuple[EvalItem, ...]
    documents: tuple[Document, ...]
    spans: Mapping[str, tuple[ResolvedSpan, ...]] = field(default_factory=dict)

    @property
    def answerable(self) -> tuple[EvalItem, ...]:
        """The items with at least one gold span."""
        return tuple(item for item in self.items if item.answerable)

    @property
    def unanswerable(self) -> tuple[EvalItem, ...]:
        """The items with no gold span."""
        return tuple(item for item in self.items if not item.answerable)


def load_items(path: str | Path) -> tuple[EvalItem, ...]:
    """Read ``dev.jsonl`` (see the module docstring for the item schema).

    Raises
    ------
    EvaluationError
        The file is missing, a line is not a JSON object, a required key is
        missing or mistyped, or an id repeats.

    """
    path = Path(path)
    if not path.is_file():
        raise EvaluationError(f"no such dataset file: {path}", path=str(path))
    items: list[EvalItem] = []
    seen: set[str] = set()
    for number, raw in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not raw.strip():
            continue
        try:
            data = json.loads(raw)
        except json.JSONDecodeError as exc:
            raise EvaluationError(
                f"{path.name}:{number}: not valid JSON ({exc.msg})", line=number
            ) from None
        item = _parse_item(data, f"{path.name}:{number}")
        if item.id in seen:
            raise EvaluationError(
                f"{path.name}:{number}: duplicate item id {item.id!r}", line=number
            )
        seen.add(item.id)
        items.append(item)
    if not items:
        raise EvaluationError(f"{path.name} holds no items", path=str(path))
    return tuple(items)


def _parse_item(data: Any, where: str) -> EvalItem:
    if not isinstance(data, dict):
        raise EvaluationError(f"{where}: expected a JSON object")
    unknown = set(data) - {"id", "question", "gold", "answer", "tags"}
    if unknown:
        raise EvaluationError(f"{where}: unknown keys {sorted(unknown)}")
    try:
        gold_raw = data.get("gold", [])
        if not isinstance(gold_raw, list):
            raise TypeError("gold must be a list")
        gold = tuple(
            GoldSpan(source_uri=str(g["source_uri"]), quote=str(g["quote"]))
            for g in gold_raw
        )
        answer = data.get("answer")
        if answer is not None and not isinstance(answer, str):
            raise TypeError("answer must be a string")
        tags = data.get("tags", [])
        if not isinstance(tags, list) or not all(isinstance(t, str) for t in tags):
            raise TypeError("tags must be a list of strings")
        return EvalItem(
            id=str(data["id"]),
            question=str(data["question"]),
            gold=gold,
            answer=answer,
            tags=tuple(tags),
        )
    except (KeyError, TypeError, ValueError) as exc:
        raise EvaluationError(f"{where}: {exc}") from None


def load_corpus(corpus_dir: str | Path, *, glob: str = "**/*") -> tuple[Document, ...]:
    """Load every file under *corpus_dir* and normalise its text.

    Raises
    ------
    EvaluationError
        *corpus_dir* is not a directory, a file failed to load, or nothing
        loaded — a dataset corpus is frozen and must load completely.

    """
    corpus_dir = Path(corpus_dir)
    if not corpus_dir.is_dir():
        raise EvaluationError(f"no such corpus directory: {corpus_dir}")
    report = DirectoryLoader().load_path(corpus_dir, glob=glob)
    if report.failed:
        failed = ", ".join(i.source_uri for i in report.failed)
        raise EvaluationError(f"corpus files failed to load: {failed}")
    if not report.loaded:
        raise EvaluationError(f"corpus directory holds no documents: {corpus_dir}")
    return tuple(
        Document(
            doc_id=d.doc_id,
            source_uri=d.source_uri,
            text=normalize_text(d.text),
            content_hash=d.content_hash,
            metadata=d.metadata,
        )
        for d in sorted(report.loaded, key=lambda d: d.source_uri)
    )


def resolve_spans(
    items: Sequence[EvalItem], documents: Sequence[Document]
) -> dict[str, tuple[ResolvedSpan, ...]]:
    """Locate every item's gold quotes in the corpus.

    A quote is matched with runs of whitespace collapsed, so it may span a
    line wrap or a CRLF/LF difference, and must occur **exactly once** in
    its document.

    Raises
    ------
    EvaluationError
        A gold ``source_uri`` is not in the corpus, or a quote is missing or
        ambiguous. The message names the item.

    """
    by_uri = {d.source_uri: d for d in documents}
    resolved: dict[str, tuple[ResolvedSpan, ...]] = {}
    for item in items:
        spans = []
        for gold in item.gold:
            document = by_uri.get(gold.source_uri)
            if document is None:
                raise EvaluationError(
                    f"item {item.id!r}: gold source {gold.source_uri!r} is not in "
                    "the corpus",
                    item=item.id,
                )
            pattern = r"\s+".join(re.escape(word) for word in gold.quote.split())
            matches = list(re.finditer(pattern, document.text))
            if len(matches) != 1:
                what = "not found" if not matches else f"found {len(matches)} times"
                raise EvaluationError(
                    f"item {item.id!r}: quote {what} in {gold.source_uri}: "
                    f"{gold.quote[:60]!r}",
                    item=item.id,
                )
            spans.append(
                ResolvedSpan(
                    doc_id=document.doc_id,
                    source_uri=document.source_uri,
                    start_char=matches[0].start(),
                    end_char=matches[0].end(),
                )
            )
        resolved[item.id] = tuple(spans)
    return resolved


def load_dataset(
    items_path: str | Path, corpus_dir: str | Path | None = None
) -> EvalDataset:
    """Load items + corpus and resolve every gold span.

    Parameters
    ----------
    items_path
        The ``dev.jsonl`` file.
    corpus_dir
        Defaults to ``corpus/`` next to *items_path*.

    """
    items_path = Path(items_path)
    corpus = (
        Path(corpus_dir) if corpus_dir is not None else items_path.parent / "corpus"
    )
    items = load_items(items_path)
    documents = load_corpus(corpus)
    return EvalDataset(
        name=items_path.parent.name,
        items=items,
        documents=documents,
        spans=resolve_spans(items, documents),
    )
