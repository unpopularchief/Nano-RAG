"""``LocalCrossEncoderReranker`` — offline reranking via a ``fastembed`` ONNX model.

No network after the first model download, no rate limiting, no API key —
the same offline-by-default property the local embedder gives ingest
(plan.md §4), now for reranking. plan.md §9 Phase F Acceptance: "The local
cross-encoder is preferred over the Jina API if it matches within noise —
it keeps the pipeline offline."

``fastembed`` is imported lazily, inside ``__init__`` — it is already the
``[local]`` extra's dependency (no new one added here), but importing
``nanorag.rerank`` still must not require it. Missing the extra raises
``ConfigError`` naming the install command, matching
``FastEmbedEmbedder``'s own boundary.

Measured (``docs/evaluation.md`` "Reranking") to beat the Phase D
baseline by a wide margin — but **not** wired into ``Rag.from_defaults()``:
construction here eagerly loads the ONNX model, and ``from_defaults()``
also builds the ``Rag`` behind ``nanorag ingest``, which plan.md §4 says
must stay network-free. Wiring this in safely needs lazy reranker
construction (load on first ``retrieve()``/``query()``, not at
``Rag()``/``from_defaults()`` time) — a named follow-up, not bundled into
F1. Pass it explicitly today: ``Rag.from_defaults(reranker=
LocalCrossEncoderReranker(), retrieve_k=32)``.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from nanorag.errors import ConfigError, RerankError
from nanorag.types import ScoredChunk

if TYPE_CHECKING:
    from fastembed.rerank.cross_encoder import TextCrossEncoder

#: Small (~80 MB), fast, English-only — a deliberately light default given
#: the 8 GB VRAM ceiling is reserved for generation (plan.md §4).
DEFAULT_MODEL_ID = "Xenova/ms-marco-MiniLM-L-6-v2"

#: The ``ScoredChunk.source`` a successful rerank stamps (plan.md §5).
SOURCE = "rerank:local"


class LocalCrossEncoderReranker:
    """Offline cross-encoder reranking via a ``fastembed`` ONNX model.

    Parameters
    ----------
    model_id
        A ``fastembed`` cross-encoder model name.
    cache_dir
        Where ``fastembed`` caches downloaded model files.
    threads
        ONNX Runtime thread count; ``None`` uses ``fastembed``'s default.

    Raises
    ------
    ConfigError
        ``fastembed``/``onnxruntime`` are not installed, or *model_id* is
        not a cross-encoder model ``fastembed`` supports.

    """

    def __init__(
        self,
        model_id: str = DEFAULT_MODEL_ID,
        *,
        cache_dir: str | None = None,
        threads: int | None = None,
    ) -> None:
        """Load *model_id*'s ONNX cross-encoder (see the class docstring)."""
        try:
            from fastembed.rerank.cross_encoder import TextCrossEncoder
        except ImportError as exc:
            raise ConfigError(
                "local reranking needs the [local] extra: run "
                '`uv add "nanorag[local]"` (installs fastembed + onnxruntime)'
            ) from exc

        self.model_id = model_id
        try:
            self._model: TextCrossEncoder = TextCrossEncoder(
                model_name=model_id, cache_dir=cache_dir, threads=threads
            )
        except ValueError as exc:
            raise ConfigError(
                f"{model_id!r} is not a fastembed-supported cross-encoder model",
                model_id=model_id,
            ) from exc

    def rerank(
        self, query: str, hits: list[ScoredChunk], top_n: int
    ) -> list[ScoredChunk]:
        """Re-score *hits* against *query* and return the best ``top_n``.

        Raises
        ------
        RerankError
            The model failed to score the batch.

        """
        if not hits:
            return []
        try:
            scores = list(self._model.rerank(query, [h.chunk.text for h in hits]))
        except Exception as exc:
            # fastembed/onnxruntime raise a variety of undocumented exception
            # types across versions and backends; normalise all of them at
            # this boundary, exactly as FastEmbedEmbedder does for embedding.
            raise RerankError(f"local reranking failed: {exc}") from exc
        scored = [
            ScoredChunk(chunk=hit.chunk, score=float(score), source=SOURCE)
            for hit, score in zip(hits, scores, strict=True)
        ]
        scored.sort(key=lambda h: h.score, reverse=True)
        return scored[:top_n]
