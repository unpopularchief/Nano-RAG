"""``Rag`` — the facade: ~30 lines of composition, no logic of its own.

The two rules that keep this from becoming a framework (plan.md §5):

1. **The facade owns no logic.** ``Rag.query()`` is a readable composition
   of public functions — retrieve, rerank, budget, build context, build
   prompt, generate — every one of which a caller can invoke directly. The
   pieces of arithmetic it needs (:func:`context_budget`,
   :func:`insufficient_context`, :func:`load_vectors`, :func:`rerank_hits`)
   are module-level functions here, not methods, for the same reason — as
   are the three ``from_defaults()`` choices (:func:`default_embedder`,
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
from dataclasses import replace
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np

from nanorag.chunking.base import Chunker
from nanorag.chunking.recursive import RecursiveChunker
from nanorag.citations.parser import CitationReport, parse_citations
from nanorag.config import Settings
from nanorag.context.builder import Context, ContextBuilder
from nanorag.embeddings.base import Embedder
from nanorag.errors import (
    ConfigError,
    IndexModelMismatch,
    RerankError,
    RetrievalError,
    StoreError,
)
from nanorag.generation.base import Generator
from nanorag.generation.presets import GROQ, OLLAMA
from nanorag.hashing import normalize_text
from nanorag.loaders.directory import DirectoryLoader
from nanorag.observability.timing import Timer
from nanorag.prompting.templates import INSUFFICIENT_CONTEXT_TEXT, PromptBuilder
from nanorag.rerank.base import Reranker
from nanorag.rerank.identity import IdentityReranker
from nanorag.retrieval.dense import DenseRetriever
from nanorag.store.filters import Filter
from nanorag.store.numpy_store import NumpyVectorStore
from nanorag.store.sqlite_docs import SqliteDocumentStore
from nanorag.tokens import TiktokenCounter
from nanorag.types import (
    Answer,
    Chunk,
    Document,
    IngestReport,
    ScoredChunk,
    SyncReport,
    Usage,
)

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
#: Metadata key a document opts into near-duplicate detection with (plan.md
#: §18 F13): the value is a blocking key, and only chunks sharing it are ever
#: compared against each other. A document without this key pays nothing.
DEDUP_GROUP_KEY = "nanorag.dedup_group"
#: Metadata key a losing chunk is tagged with: the winning chunk's id.
DEDUP_OF_KEY = "nanorag.duplicate_of"
#: Cosine similarity at or above which two chunks in the same dedup group
#: count as near-duplicates (plan.md §11 "hash, then cosine > 0.97") —
#: deliberately not embedder-conditional like :data:`DEFAULT_MIN_SCORE`:
#: "almost identical text" sits in the same very-high range for any sane
#: embedding model, unlike "good enough to answer from".
DEFAULT_DUPLICATE_THRESHOLD = 0.97
#: :func:`Rag.sync_path` refuses ``apply=True`` past this fraction of the
#: store's document count deleted in one run (plan.md §18 F9) — a
#: deliberately conservative catch for "wrong root" / "unmounted volume",
#: not a tuned value; override per call for a corpus with legitimate bulk
#: deletions.
DEFAULT_MAX_DELETE_FRACTION = 0.5

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


def rerank_hits(
    reranker: Reranker, query: str, hits: list[ScoredChunk], top_n: int
) -> list[ScoredChunk]:
    """Rerank *hits* to *top_n* via *reranker*, never failing the caller.

    A module-level function, not a ``Rag`` method, for the same reason
    :func:`context_budget`/:func:`insufficient_context` are (plan.md §5
    rule 1): both :meth:`Rag.retrieve` and :meth:`Rag.query` call it, and
    so does the eval runner (:func:`~nanorag.evaluation.runner.
    evaluate_retrieval`), which measures a reranker's effect on *ranking*
    the same way ``Rag`` itself would, wired from public pieces
    (``rag.reranker``) rather than duplicating this fallback logic.

    A ``reranker`` that raises ``RerankError`` degrades to the retriever's
    own order (``hits[:top_n]``), with a warning logged, rather than
    failing the call (plan.md §9 Phase F Tests).
    """
    try:
        return reranker.rerank(query, hits, top_n)
    except RerankError:
        log.warning(
            "reranker %r failed, falling back to retriever order",
            reranker,
            exc_info=True,
        )
        return hits[:top_n]


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
        Chunks kept per query, after reranking.
    reranker
        Re-scores the ``retrieve_k`` candidates down to *k* (plan.md §9
        Phase F). Defaults to :class:`~nanorag.rerank.identity.
        IdentityReranker` — a pure slice, no re-scoring — so a caller who
        never touches this parameter sees behaviour identical to before
        Phase F. A reranker that raises ``RerankError`` degrades the query
        to the retriever's own order (a warning is logged) rather than
        failing it.
    retrieve_k
        Candidates the retriever fetches *before* reranking. Defaults to
        *k* itself — with the identity reranker this makes "retrieve" and
        "keep" the same set, exactly the pre-Phase-F behaviour. Pass a
        larger value when *reranker* is a real one, so it has more than
        *k* candidates to choose from (plan.md §9 Phase F "retrieve-k /
        rerank-to-n wiring").
    budget_margin
        Safety fraction taken off the context budget.
    min_results, min_gap, min_score
        :func:`insufficient_context` thresholds. ``min_score`` is on the
        embedder's own score scale, so it defaults to off here;
        :meth:`from_defaults` sets :data:`DEFAULT_MIN_SCORE` for the
        default embedder.
    duplicate_threshold
        Cosine similarity at or above which :meth:`ingest` flags two
        chunks in the same :data:`DEDUP_GROUP_KEY` group as near-duplicates
        (plan.md §18 F13). Defaults to :data:`DEFAULT_DUPLICATE_THRESHOLD`.

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
        reranker: Reranker | None = None,
        retrieve_k: int | None = None,
        budget_margin: float = DEFAULT_BUDGET_MARGIN,
        min_results: int = 1,
        min_gap: float = 0.0,
        min_score: float | None = None,
        duplicate_threshold: float = DEFAULT_DUPLICATE_THRESHOLD,
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
        self.k = k
        self.retrieve_k = retrieve_k if retrieve_k is not None else k
        self.retriever = DenseRetriever(embedder, self.vectors, docs, k=self.retrieve_k)
        self.reranker: Reranker = (
            reranker if reranker is not None else IdentityReranker()
        )
        self.budget_margin = budget_margin
        self.min_results = min_results
        self.min_gap = min_gap
        self.min_score = min_score
        self.duplicate_threshold = duplicate_threshold

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

        A document whose ``content_hash`` matches what is already stored
        for its ``doc_id`` is skipped entirely — no chunking, no embedder
        call, no store write (plan.md §9 Phase E "change detection"). This
        is a document-level shortcut only: it never substitutes for the
        chunk-level reuse a *changed* document still gets from
        ``CachingEmbedder`` (keyed by each chunk's own text, not this
        hash), which is what keeps re-embedding a large edited document
        cheap (plan.md §5).

        Otherwise, the text is put through :func:`~nanorag.hashing.
        normalize_text` (NFC + LF) before chunking — a ``Loader`` returns
        raw bytes-as-read untouched (plan.md §6), so without this step a
        CRLF checkout and an LF checkout of the identical content chunk
        differently and retrieve differently. ``content_hash`` needs no
        adjustment: it already runs the same normalisation internally, so
        it is unchanged by this rewrite (``normalize_text`` is idempotent).
        A chunk in a document opted into near-duplicate detection (see
        :meth:`_tag_near_duplicates`) may be tagged before it is written.
        The chunks then replace any prior version's in one SQLite
        transaction, their vectors are persisted, and only then is the
        in-memory index updated — rows for chunks that no longer exist are
        dropped from it, new ones upserted (plan.md §18 F2 ordering).
        """
        total = 0
        for document in documents:
            existing = self.docs.get_document(document.doc_id)
            if existing is not None and existing.content_hash == document.content_hash:
                total += self.docs.count_chunks(document.doc_id)
                continue
            normalized = normalize_text(document.text)
            if normalized != document.text:
                document = replace(document, text=normalized)
            stale = {c.chunk_id for c in self.docs.get_chunks(document.doc_id)}
            chunks = self.chunker.chunk(document)
            ids = [c.chunk_id for c in chunks]
            vectors: np.ndarray | None = None
            if chunks:
                vectors = self.embedder.embed([c.text for c in chunks])
                chunks = self._tag_near_duplicates(
                    document, chunks, vectors, exclude=stale | set(ids)
                )
            self.docs.upsert_document(document, chunks)
            if chunks and vectors is not None:
                self.docs.upsert_embeddings(
                    self.embedder.model_id, self.embedder.dim, ids, vectors
                )
            gone = sorted(stale - set(ids))
            if gone:
                self.vectors.delete(gone)
            if chunks and vectors is not None:
                self.vectors.upsert(ids, vectors)
            total += len(chunks)
        return total

    def _tag_near_duplicates(
        self,
        document: Document,
        chunks: Sequence[Chunk],
        vectors: np.ndarray,
        *,
        exclude: set[str],
    ) -> list[Chunk]:
        """Flag chunks that near-duplicate an already-indexed one (plan.md §18 F13).

        Bounded by a blocking key so this never compares against the whole
        corpus, the O(n)-per-chunk cost F13 warns against: a document is
        checked at all only if it sets ``metadata[DEDUP_GROUP_KEY]``, and
        then only against chunks sharing that exact group (found via
        ``docs.filter_chunk_ids`` — the metadata filter already used for
        retrieval pre-filtering, not a new index). A corpus that never sets
        the key pays nothing.

        A chunk found at or above ``duplicate_threshold`` cosine similarity
        to a group member from a different document — *different*
        enforced by *exclude*, this document's own current and prior chunk
        ids — is the loser: kept, never dropped (dropping would rotate
        every later chunk's ordinal and id for no reason, and break any
        citation already pointing at it), but tagged
        ``{DEDUP_OF_KEY: winner_chunk_id}`` in its own metadata.

        Comparisons run against the index *before* this document's chunks
        are upserted into it, so two never-before-seen near-duplicate
        documents ingested in the same batch do not catch each other —
        only a duplicate of something already persisted. Documented, not
        fixed: catching batch-internal duplicates needs an all-pairs
        comparison within the batch, which is unbounded in the same way
        F13 warns against for the corpus as a whole.
        """
        group = document.metadata.get(DEDUP_GROUP_KEY)
        if group is None:
            return list(chunks)
        candidates = self.docs.filter_chunk_ids({DEDUP_GROUP_KEY: group}) - exclude
        if not candidates:
            return list(chunks)
        tagged = []
        for chunk, vector in zip(chunks, vectors, strict=True):
            hits = self.vectors.search(vector, k=1, allowed_ids=candidates)
            if hits and hits[0][1] >= self.duplicate_threshold:
                chunk = replace(
                    chunk, metadata={**chunk.metadata, DEDUP_OF_KEY: hits[0][0]}
                )
            tagged.append(chunk)
        return tagged

    def delete_document(self, doc_id: str) -> int:
        """Remove a document from search and the store; return chunks removed.

        Cascades through SQLite (chunks, then embeddings — ``ON DELETE
        CASCADE``) and tombstones the same chunk ids in the in-memory index
        in the same call, so a deleted chunk is unreachable from both
        ``retrieve()`` and a fresh :func:`load_vectors` rebuild from the
        same instant (plan.md §9 Phase E). The rows are not yet physically
        reclaimed in the index — call :meth:`compact` for that. An unknown
        *doc_id* is a no-op.
        """
        chunk_ids = [c.chunk_id for c in self.docs.get_chunks(doc_id)]
        self.docs.delete_document(doc_id)
        if chunk_ids:
            self.vectors.delete(chunk_ids)
        return len(chunk_ids)

    def compact(self) -> None:
        """Physically reclaim tombstoned rows in the in-memory index.

        Deletion alone (:meth:`delete_document`) already excludes tombstoned
        chunks from search; this only reclaims their space (plan.md §6).
        """
        self.vectors.compact()

    def sync_path(
        self,
        root: str | Path,
        *,
        glob: str = "**/*",
        ignore: Sequence[str] = (),
        apply: bool = False,
        max_delete_fraction: float = DEFAULT_MAX_DELETE_FRACTION,
    ) -> SyncReport:
        """Reconcile the store against *root*: add, update, delete.

        Dry run by default (``apply=False``): walks *root* exactly as
        :meth:`ingest_path` would and returns the plan without writing
        anything. Comparing the walk's ``source_uri`` values against the
        store's current documents classifies each into ``added`` (on disk,
        not yet stored), ``updated``/``unchanged`` (in both, split by
        ``content_hash``) or ``deleted`` (stored, no longer produced by the
        walk). Pass ``apply=True`` to execute the plan: ``added`` and
        ``updated`` go through :meth:`ingest` (whose own change detection
        is what makes a re-sync of 1,000 mostly-unchanged documents cheap —
        plan.md §9 Phase E), and ``deleted`` through :meth:`delete_document`.

        Parameters
        ----------
        root
            Directory to walk (see :meth:`ingest_path`).
        glob, ignore
            Same semantics as :meth:`ingest_path` / ``DirectoryLoader``.
        apply
            ``False`` (default): compute and return the plan only.
            ``True``: execute it.
        max_delete_fraction
            ``apply=True`` refuses to run if ``len(deleted)`` exceeds this
            fraction of the store's document count from before the sync —
            a mistyped *root* or an unmounted volume must not silently
            empty the corpus (plan.md §18 F9). Checked, but never raised,
            on a dry run: :attr:`SyncReport.over_delete_guard` reports the
            same condition either way.

        Raises
        ------
        ConfigError
            *max_delete_fraction* is not in ``[0, 1]``.
        StoreError
            ``apply=True`` and the deletions would exceed
            *max_delete_fraction*. Nothing is written when this is raised
            — ``added``/``updated`` are not applied either, so a sync
            either fully succeeds or fully does not.

        """
        if not 0 <= max_delete_fraction <= 1:
            raise ConfigError(
                f"max_delete_fraction must be in [0, 1], got {max_delete_fraction}"
            )
        report = self.loader.load_path(root, glob=glob, ignore=ignore)
        on_disk = {d.source_uri: d for d in report.loaded}
        stored = {d.source_uri: d.doc_id for d in self.docs.iter_documents()}

        added = sorted(on_disk.keys() - stored.keys())
        updated = []
        unchanged = []
        for uri in sorted(on_disk.keys() & stored.keys()):
            current = self.docs.get_document(stored[uri])
            if (
                current is not None
                and current.content_hash == on_disk[uri].content_hash
            ):
                unchanged.append(uri)
            else:
                updated.append(uri)
        deleted = sorted(stored.keys() - on_disk.keys())

        corpus_size = self.docs.count_documents()
        over_guard = len(deleted) > max_delete_fraction * corpus_size

        if apply:
            if over_guard:
                raise StoreError(
                    "sync would delete more than max_delete_fraction of the "
                    "corpus; pass a higher max_delete_fraction if this is "
                    "really intended",
                    to_delete=len(deleted),
                    corpus_size=corpus_size,
                    max_delete_fraction=max_delete_fraction,
                )
            self.ingest(on_disk[uri] for uri in added + updated)
            for uri in deleted:
                self.delete_document(stored[uri])

        return SyncReport(
            added=tuple(added),
            updated=tuple(updated),
            unchanged=tuple(unchanged),
            deleted=tuple(deleted),
            skipped=report.skipped,
            failed=report.failed,
            applied=apply,
            over_delete_guard=over_guard,
        )

    # -- read path -------------------------------------------------------------

    def retrieve(
        self, query: str, k: int | None = None, filter: Filter | None = None
    ) -> list[ScoredChunk]:
        """Top-*k* chunks for *query*: retrieve, then rerank down to *k*.

        Fetches ``max(self.retrieve_k, k)`` pre-filtered candidates (see
        ``DenseRetriever``) and reranks them to *k* via ``self.reranker`` —
        the identity reranker by default, a pure slice that makes this
        byte-identical to retrieval before Phase F unless a real reranker
        is wired in. A reranker that raises ``RerankError`` degrades to
        the retriever's own order, with a warning logged, rather than
        failing the call (plan.md §9 Phase F Tests).
        """
        final_k = self._resolve_k(k)
        hits = self.retriever.retrieve(
            query, k=max(self.retrieve_k, final_k), filter=filter
        )
        return rerank_hits(self.reranker, query, hits, final_k)

    def _resolve_k(self, k: int | None) -> int:
        """Return *k*, or ``self.k``, validated ``>= 1``."""
        final_k = k if k is not None else self.k
        if final_k < 1:
            raise RetrievalError(f"k must be >= 1, got {final_k}")
        return final_k

    def query(
        self, question: str, *, k: int | None = None, filter: Filter | None = None
    ) -> Answer:
        """Retrieve, budget, fence, generate — and say so if there is nothing.

        Every step is a public function a caller could run by hand; this
        method only sequences them and records timings. Abstention
        (``insufficient_context``) is judged on the retriever's own dense
        hits, never the reranked ones: its thresholds live on the
        embedder's score scale (plan.md §18 F10, ``docs/conventions.md``
        "Abstention"), and a reranker's score generally is not — Jina's and
        the local cross-encoder's are both on their own, uncalibrated
        scales. Reranking still decides what goes *in* the prompt; it just
        never gets to decide whether there was enough to prompt with.
        """
        timer = Timer()
        final_k = self._resolve_k(k)
        with timer.stage("retrieve"):
            dense_hits = self.retriever.retrieve(
                question, k=max(self.retrieve_k, final_k), filter=filter
            )
            hits = rerank_hits(self.reranker, question, dense_hits, final_k)

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
                dense_hits[:final_k],
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
