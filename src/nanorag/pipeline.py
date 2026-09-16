"""``Rag`` — the facade: ~30 lines of composition, no logic of its own.

The two rules that keep this from becoming a framework (plan.md §5):

1. **The facade owns no logic.** ``Rag.query()`` is a readable composition
   of public functions — retrieve, budget, build context, build prompt,
   generate — every one of which a caller can invoke directly. The three
   pieces of arithmetic it needs (:func:`context_budget`,
   :func:`insufficient_context`, :func:`load_vectors`) are module-level
   functions here, not methods, for the same reason — as are the three
   ``from_defaults()`` choices (:func:`default_embedder`,
   :func:`default_min_score`, :func:`default_generator`), so an eval run
   or a CLI command can wire exactly what a user would get.
2. **Protocols, not base classes.** Every component is duck-typed:
   ``Rag(embedder=..., generator=..., docs=...)`` accepts anything with the
   right shape.

Read path, in order (plan.md §5 diagram): retriever (embed + pre-filtered
exact search) → context builder (budget, dedup, ``[n]`` labels) → prompt
builder (nonce fence) → generator (one metered call) → ``Answer``. Zero
context blocks short-circuit *before* the generator: the answer is the
fixed :data:`~nanorag.prompting.INSUFFICIENT_CONTEXT_TEXT` with
``insufficient_context=True`` and no model call — the engine says "I don't
know" rather than inventing something (plan.md §11).

Write path: loader → chunker → embedder → document store → vector store,
committing SQLite **before** touching the in-memory matrix (plan.md §18 F2).

Citation resolution: ``nanorag.citations.parse_citations`` runs over the
generator's raw text against the same ``Context`` it was prompted with, and
``Answer.citations`` is its resolved, label-ordered result.
``Answer.contexts`` holds exactly the chunks that were in the prompt so a
reader can see the sources regardless of what the model chose to cite.
"""

from __future__ import annotations

import logging
import os
from collections.abc import Callable, Iterable, Sequence
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np

from nanorag.chunking.base import Chunker
from nanorag.chunking.recursive import RecursiveChunker
from nanorag.citations.parser import CitationReport, parse_citations
from nanorag.config import Settings
from nanorag.context.builder import Context, ContextBuilder
from nanorag.embeddings.base import Embedder
from nanorag.errors import ConfigError, IndexModelMismatch
from nanorag.generation.base import Generator
from nanorag.generation.presets import GROQ, OLLAMA
from nanorag.loaders.directory import DirectoryLoader
from nanorag.observability.timing import Timer
from nanorag.prompting.templates import INSUFFICIENT_CONTEXT_TEXT, PromptBuilder
from nanorag.retrieval.dense import DenseRetriever
from nanorag.store.filters import Filter
from nanorag.store.numpy_store import NumpyVectorStore
from nanorag.store.sqlite_docs import SqliteDocumentStore
from nanorag.tokens import TiktokenCounter
from nanorag.types import Answer, Document, IngestReport, ScoredChunk, Usage

if TYPE_CHECKING:
    from nanorag.tokens import TokenCounter

log = logging.getLogger("nanorag")

#: Default number of chunks retrieved per query.
DEFAULT_K = 8
#: Safety margin on the context budget (plan.md §11, §18 F4): the token
#: counter is an approximation of the generator's real tokenizer.
DEFAULT_BUDGET_MARGIN = 0.15
#: ``Usage.provider`` on an answer that never reached a generator.
NO_PROVIDER = "none"
#: :func:`insufficient_context` floor that ``from_defaults()`` applies for the
#: default embedder, ``bge-base-en-v1.5`` — measured at Gate C (unrelated
#: questions topped out at 0.60, answerable ones started at 0.61; see the
#: function docstring). Meaningless for any other embedder, so ``Rag()``
#: itself defaults to no floor.
DEFAULT_MIN_SCORE = 0.55

_EMPTY_CONTEXT = Context(
    blocks=(), labels={}, text="", token_count=0, budget_tokens=1, truncated=False
)


# --- the public functions the facade composes ---------------------------------


def context_budget(
    context_window: int,
    max_output_tokens: int,
    overhead_tokens: int,
    *,
    margin: float = DEFAULT_BUDGET_MARGIN,
) -> int:
    """Tokens available for retrieved context in one prompt.

    ``(window − reserved output − prompt overhead) × (1 − margin)``, floored
    at 1 (plan.md §11 "Context limits"). The window and reservation come
    from the generator, never a constant; the overhead is everything in the
    prompt that is not context (system prompt, fence lines, question).

    Raises
    ------
    ConfigError
        *margin* is not in ``[0, 1)``.

    """
    if not 0 <= margin < 1:
        raise ConfigError(f"margin must be in [0, 1), got {margin}")
    free = context_window - max_output_tokens - overhead_tokens
    return max(1, int(free * (1 - margin)))


def insufficient_context(
    hits: Sequence[ScoredChunk],
    *,
    min_results: int = 1,
    min_gap: float = 0.0,
    min_score: float | None = None,
) -> bool:
    """Return True when retrieval found nothing worth answering from.

    A tunable heuristic, not a guarantee (plan.md §18 F10). Three checks,
    any of which flags the query; the last two are off by default:

    - fewer than *min_results* hits;
    - *min_score*: the top score is below this absolute floor;
    - *min_gap*: with two or more hits, the top score stands out from the
      mean of the rest by less than this — nothing was clearly a match.

    **Locked at Gate C by measurement** (``bge-base-en-v1.5``, two corpora
    of ~120 chunks, 33 answerable vs 32 unrelated questions, ``k=8``):

    - The relative gap (top-1 − mean of the rest) does **not** separate.
      Answerable questions on a corpus that covers their topic densely
      have many near-equal hits (gap median 0.05–0.07, min 0.005);
      unrelated questions score uniformly low with the same small gap
      (median 0.015, max 0.04). Any threshold that catches most unrelated
      questions abstains on 20–40 % of answerable ones. So ``min_gap``
      ships disabled and stays available for corpora where it does help.
    - The absolute top-1 score does separate, but on a scale that is the
      embedder's, not a constant: with ``bge-base-en-v1.5`` every
      answerable question scored ≥ 0.61 and every unrelated one ≤ 0.60
      (median 0.48); on-topic-but-uncovered questions land in 0.55–0.71,
      which is the model's job to abstain on, not retrieval's. ``min_score``
      ships disabled because ``Rag`` takes any embedder; **0.55 is the
      measured setting for the default embedder** and is the one
      ``from_defaults()`` uses (see :data:`DEFAULT_MIN_SCORE`).

    ``Rag.query()`` reports the outcome on ``Answer.insufficient_context``
    and still asks the model, which the prompt instructs to abstain; only
    an empty context skips the call.
    """
    if len(hits) < min_results:
        return True
    if min_score is not None and hits and hits[0].score < min_score:
        return True
    if min_gap > 0 and len(hits) >= 2:
        rest = [h.score for h in hits[1:]]
        return hits[0].score - sum(rest) / len(rest) < min_gap
    return False


def load_vectors(docs: SqliteDocumentStore, dim: int) -> NumpyVectorStore:
    """Rebuild the in-memory index from the embeddings persisted in *docs*."""
    vectors = NumpyVectorStore(dim=dim)
    ids: list[str] = []
    rows: list[np.ndarray] = []
    for chunk_id, vector in docs.iter_embeddings():
        ids.append(chunk_id)
        rows.append(vector)
    if ids:
        vectors.upsert(ids, np.stack(rows))
    return vectors


def default_embedder(settings: Settings, root: str | Path) -> Embedder:
    """Build the embedder ``from_defaults()`` wires: local ONNX, batched, cached.

    ``FastEmbedEmbedder(settings.embedding_model)`` inside a
    ``BatchingEmbedder`` (``settings.embed_batch_size``) inside a
    ``CachingEmbedder`` whose cache lives at ``root/embeddings.sqlite`` —
    so a re-ingest, or an eval sweep over chunk sizes, re-embeds only text
    it has never seen (plan.md §5).

    Raises
    ------
    ConfigError
        The ``[local]`` extra is missing, or the model id is unknown to
        ``fastembed`` — each names the fix.

    """
    from nanorag.embeddings.batching import BatchingEmbedder
    from nanorag.embeddings.cache import CachingEmbedder, EmbeddingCache
    from nanorag.embeddings.local import FastEmbedEmbedder

    local = FastEmbedEmbedder(settings.embedding_model)
    return CachingEmbedder(
        BatchingEmbedder(local, batch_size=settings.embed_batch_size),
        EmbeddingCache(Path(root) / "embeddings.sqlite"),
    )


def default_min_score(embedding_model: str) -> float | None:
    """Return the :func:`insufficient_context` floor measured for *embedding_model*.

    :data:`DEFAULT_MIN_SCORE` for the default embedder, else ``None`` — the
    scale is the embedder's, and no other model has been measured.
    """
    from nanorag.embeddings.local import DEFAULT_MODEL_ID

    return DEFAULT_MIN_SCORE if embedding_model == DEFAULT_MODEL_ID else None


def default_generator(
    settings: Settings,
    *,
    env: Callable[[str], str | None] = os.environ.get,
    ollama_reachable: Callable[[str], bool] | None = None,
) -> Generator:
    """Pick the generator(s) by what is available (plan.md §7).

    ``generator_preset="auto"`` chains every available option in order —
    Groq key → Groq, Gemini key → Gemini, a reachable Ollama → Ollama — as a
    ``FallbackGenerator`` (the Gate C recommendation: a silent daily-cap
    failure is the exact trap fallback exists for). A named preset uses
    only that one. The choice is logged at ``INFO``.

    Raises
    ------
    ConfigError
        Nothing is available; the message names all three options.

    """
    from nanorag.generation.fallback import FallbackGenerator
    from nanorag.generation.gemini import API_KEY_ENV, GeminiGenerator
    from nanorag.generation.openai_compat import OpenAICompatGenerator

    if ollama_reachable is None:
        ollama_reachable = _ollama_reachable
    want = settings.generator_preset
    chosen: list[Generator] = []
    groq_key = env(GROQ.api_key_env or "")
    if want in ("auto", "groq") and groq_key:
        chosen.append(
            OpenAICompatGenerator(
                GROQ,
                api_key=groq_key,
                timeout=settings.request_timeout_s,
                **_retry_kwargs(settings),
            )
        )
    gemini_key = env(API_KEY_ENV)
    if want in ("auto", "gemini") and gemini_key:
        chosen.append(
            GeminiGenerator(
                api_key=gemini_key,
                timeout=settings.request_timeout_s,
                **_retry_kwargs(settings),
            )
        )
    if want in ("auto", "ollama") and ollama_reachable(OLLAMA.base_url):
        chosen.append(
            OpenAICompatGenerator(
                OLLAMA, timeout=settings.request_timeout_s, **_retry_kwargs(settings)
            )
        )
    if not chosen:
        raise ConfigError(
            "no generator available: set GROQ_API_KEY (free at console.groq.com), "
            "or GEMINI_API_KEY (free at aistudio.google.com), or run Ollama "
            f"locally at {OLLAMA.base_url} (ollama.com)",
            generator_preset=want,
        )
    log.info("generator: %s", " -> ".join(g.provider for g in chosen))
    if len(chosen) == 1:
        return chosen[0]
    return FallbackGenerator(*chosen)


def _retry_kwargs(settings: Settings) -> dict[str, Any]:
    from nanorag.ratelimit import Backoff

    return {
        "connect_timeout": settings.connect_timeout_s,
        "backoff": Backoff(
            max_retries=settings.max_retries, base_delay=settings.retry_base_delay_s
        ),
    }


def _ollama_reachable(base_url: str, timeout: float = 1.0) -> bool:
    """Return True if an OpenAI-compatible server answers ``GET {base_url}/models``."""
    import httpx

    try:
        return httpx.get(f"{base_url}/models", timeout=timeout).status_code == 200
    except httpx.HTTPError:
        return False


# --- the facade ---------------------------------------------------------------


class Rag:
    """Ingest documents; answer questions with cited context.

    Parameters
    ----------
    embedder
        Any :class:`~nanorag.embeddings.base.Embedder`.
    generator
        Any :class:`~nanorag.generation.base.Generator`.
    docs
        The document store. Its recorded ``(model_id, dim)`` must match
        *embedder*.
    vectors
        The in-memory index. Rebuilt from *docs* when omitted.
    chunker, loader, prompt_builder
        Defaults: ``RecursiveChunker()``, ``DirectoryLoader()``,
        ``PromptBuilder()``.
    counter
        Counts tokens for the context budget. Defaults to
        ``TiktokenCounter("cl100k_base")`` — an approximation for the
        Llama/Gemini/Qwen tokenizers actually in play, which is why
        *budget_margin* exists (plan.md §18 F4).
    k
        Chunks retrieved per query.
    budget_margin
        Safety fraction taken off the context budget.
    min_results, min_gap, min_score
        :func:`insufficient_context` thresholds. ``min_score`` is on the
        embedder's own score scale, so it defaults to off here;
        :meth:`from_defaults` sets :data:`DEFAULT_MIN_SCORE` for the
        default embedder.

    Raises
    ------
    IndexModelMismatch
        *docs* was built with a different embedding model or dimension.

    """

    def __init__(
        self,
        *,
        embedder: Embedder,
        generator: Generator,
        docs: SqliteDocumentStore,
        vectors: NumpyVectorStore | None = None,
        chunker: Chunker | None = None,
        loader: DirectoryLoader | None = None,
        prompt_builder: PromptBuilder | None = None,
        counter: TokenCounter | None = None,
        k: int = DEFAULT_K,
        budget_margin: float = DEFAULT_BUDGET_MARGIN,
        min_results: int = 1,
        min_gap: float = 0.0,
        min_score: float | None = None,
    ) -> None:
        """Wire the components together (see the class docstring)."""
        meta = docs.index_meta()
        if meta is not None and meta != (embedder.model_id, embedder.dim):
            raise IndexModelMismatch(
                "document store was built with a different embedding model",
                index_model=meta[0],
                index_dim=meta[1],
                embedder_model=embedder.model_id,
                embedder_dim=embedder.dim,
            )
        self.embedder = embedder
        self.generator = generator
        self.docs = docs
        self.vectors = (
            vectors if vectors is not None else load_vectors(docs, embedder.dim)
        )
        self.chunker: Chunker = chunker if chunker is not None else RecursiveChunker()
        self.loader = loader if loader is not None else DirectoryLoader()
        self.prompt_builder = (
            prompt_builder if prompt_builder is not None else PromptBuilder()
        )
        self.counter: TokenCounter = (
            counter if counter is not None else TiktokenCounter("cl100k_base")
        )
        self.retriever = DenseRetriever(embedder, self.vectors, docs, k=k)
        self.budget_margin = budget_margin
        self.min_results = min_results
        self.min_gap = min_gap
        self.min_score = min_score

    @classmethod
    def from_defaults(
        cls,
        persist_dir: str | Path | None = None,
        *,
        settings: Settings | None = None,
        generator: Generator | None = None,
        **kwargs: Any,
    ) -> Rag:
        """Local ONNX embeddings, best available generator, SQLite under *persist_dir*.

        Parameters
        ----------
        persist_dir
            Where ``nanorag.sqlite`` (documents, chunks, vectors) and
            ``embeddings.sqlite`` (the embedding cache) live. Overrides
            ``settings.persist_dir``.
        settings
            Defaults to ``Settings.load()`` (env, profile, ``pyproject``).
        generator
            Skip :func:`default_generator` and use this one.
        **kwargs
            Passed through to ``Rag(...)`` (``k``, ``counter``, …).
            ``min_score`` defaults to :data:`DEFAULT_MIN_SCORE` when the
            embedding model is the default one, else stays off.

        Raises
        ------
        ConfigError
            The ``[local]`` extra is missing, or no generator is available
            — each names the fix.

        """
        if settings is None:
            settings = Settings.load()
        root = Path(persist_dir if persist_dir is not None else settings.persist_dir)
        root.mkdir(parents=True, exist_ok=True)
        embedder = default_embedder(settings, root)
        kwargs.setdefault("min_score", default_min_score(settings.embedding_model))
        if generator is None:
            generator = default_generator(settings)
        return cls(
            embedder=embedder,
            generator=generator,
            docs=SqliteDocumentStore(root / "nanorag.sqlite"),
            **kwargs,
        )

    def close(self) -> None:
        """Close the document store (and the generator, if it can be closed)."""
        self.docs.close()
        close = getattr(self.generator, "close", None)
        if callable(close):
            close()

    # -- write path ----------------------------------------------------------

    def ingest_path(
        self, root: str | Path, *, glob: str = "**/*", ignore: Sequence[str] = ()
    ) -> IngestReport:
        """Load every matching file under *root* and ingest what loaded."""
        report = self.loader.load_path(root, glob=glob, ignore=ignore)
        self.ingest(report.loaded)
        return report

    def ingest(self, documents: Iterable[Document]) -> int:
        """Chunk, embed and persist *documents*; return the number of chunks.

        Per document: the chunks replace any prior version's in one SQLite
        transaction, their vectors are persisted, and only then is the
        in-memory index updated — rows for chunks that no longer exist are
        dropped from it, new ones upserted (plan.md §18 F2 ordering).
        """
        total = 0
        for document in documents:
            stale = {c.chunk_id for c in self.docs.get_chunks(document.doc_id)}
            chunks = self.chunker.chunk(document)
            ids = [c.chunk_id for c in chunks]
            self.docs.upsert_document(document, chunks)
            if chunks:
                vectors = self.embedder.embed([c.text for c in chunks])
                self.docs.upsert_embeddings(
                    self.embedder.model_id, self.embedder.dim, ids, vectors
                )
            gone = sorted(stale - set(ids))
            if gone:
                self.vectors.delete(gone)
            if chunks:
                self.vectors.upsert(ids, vectors)
            total += len(chunks)
        return total

    # -- read path -------------------------------------------------------------

    def retrieve(
        self, query: str, k: int | None = None, filter: Filter | None = None
    ) -> list[ScoredChunk]:
        """Top-*k* chunks for *query*, pre-filtered (see ``DenseRetriever``)."""
        return self.retriever.retrieve(query, k=k, filter=filter)

    def query(
        self, question: str, *, k: int | None = None, filter: Filter | None = None
    ) -> Answer:
        """Retrieve, budget, fence, generate — and say so if there is nothing.

        Every step is a public function a caller could run by hand; this
        method only sequences them and records timings.
        """
        timer = Timer()
        with timer.stage("retrieve"):
            hits = self.retrieve(question, k=k, filter=filter)

        with timer.stage("context"):
            overhead = self.counter.count(
                self.prompt_builder.build(question, _EMPTY_CONTEXT).as_text()
            )
            budget = context_budget(
                self.generator.context_window,
                self.generator.max_output_tokens,
                overhead,
                margin=self.budget_margin,
            )
            context = ContextBuilder(budget, counter=self.counter).build(hits)
            thin = insufficient_context(
                hits,
                min_results=self.min_results,
                min_gap=self.min_gap,
                min_score=self.min_score,
            )

        if not context.blocks:
            return Answer(
                text=INSUFFICIENT_CONTEXT_TEXT,
                citations=(),
                contexts=(),
                insufficient_context=True,
                truncated=context.truncated,
                usage=Usage(NO_PROVIDER, "", 0, 0, 0),
                timings=timer.timings(),
            )

        prompt = self.prompt_builder.build(question, context)
        with timer.stage("generate"):
            generation = self.generator.generate(prompt)
        _reconcile_usage(
            self.counter.count(prompt.as_text()),
            generation.usage.prompt_tokens,
            self.budget_margin,
        )
        report = parse_citations(
            generation.text, context, get_document=self.docs.get_document
        )
        _log_citation_issues(report)
        return Answer(
            text=generation.text,
            citations=report.citations,
            contexts=context.hits,
            insufficient_context=thin,
            truncated=context.truncated,
            usage=generation.usage,
            timings=timer.timings(),
        )


def _reconcile_usage(estimated: int, reported: int, margin: float) -> None:
    """Log when the provider counted more prompt tokens than the margin covers.

    The first half of plan.md §18 F4: the local counter is an estimate, so
    the provider's returned ``usage`` is compared against it after every
    call. A discrepancy past the margin means the budget could overflow
    the window on a larger prompt — surfaced as a warning, never silent.
    """
    if reported and estimated and reported > estimated * (1 + margin):
        log.warning(
            "provider counted %d prompt tokens, local estimate %d: over the %.0f%% "
            "margin — pass the generator's tokenizer as `counter` or raise "
            "`budget_margin`",
            reported,
            estimated,
            margin * 100,
        )


def _log_citation_issues(report: CitationReport) -> None:
    """Log when the model cited a marker that did not resolve to a block.

    The "report a validity rate" half of the citation parser's job
    (plan.md §6) — surfaced as a warning rather than a new ``Answer`` field,
    matching :func:`_reconcile_usage`'s pattern; ``Answer`` is frozen as of
    ``v0.1.0`` (plan.md §18 F11).
    """
    if report.total_markers and report.resolved_markers < report.total_markers:
        log.warning(
            "model cited %d marker(s) that did not resolve to a context block "
            "(%.0f%% of %d markers valid)",
            report.total_markers - report.resolved_markers,
            report.validity_rate * 100,
            report.total_markers,
        )
