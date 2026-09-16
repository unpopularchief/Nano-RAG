"""Chunk-size × overlap × embedding-model sweep — the Phase D3 measurement.

The sweep plan.md §9 D3 exists to enable: because embeddings are local and
unmetered, re-embedding the eval corpus for every configuration costs
nothing but CPU time, so the chunker defaults can be picked by measurement
rather than reputation. Every cell runs the same offline retrieval eval
``nanorag eval`` runs (``evaluation.evaluate_retrieval``), so the numbers
are directly comparable with the committed baseline in
``datasets/<name>/thresholds.json``.

Run from the repo root (first run downloads any model not yet cached)::

    uv run python benchmarks/eval_sweep.py
    uv run python benchmarks/eval_sweep.py --sizes 256,512 --overlaps 0,64
    uv run python benchmarks/eval_sweep.py --models BAAI/bge-base-en-v1.5

Prints one Markdown table row per configuration — paste into
``docs/evaluation.md``. The embedding cache under ``--persist-dir`` makes a
repeat of any cell free. Not part of the package and not run by the test
suite.
"""

from __future__ import annotations

import argparse
import time
from collections.abc import Sequence
from pathlib import Path

import numpy as np

from nanorag.chunking.recursive import RecursiveChunker
from nanorag.config import Settings
from nanorag.embeddings.base import Embedder
from nanorag.evaluation import build_rag, evaluate_retrieval, load_dataset
from nanorag.pipeline import default_embedder, default_min_score

COLUMNS = ("recall@1", "recall@5", "recall@10", "mrr", "ndcg@5", "false_abstain_rate")


class _QueryMemo:
    """Memoise ``embed_query`` across cells: the questions never change.

    ``CachingEmbedder`` deliberately never caches queries (they run once in
    real use); in a sweep the same 341 questions are embedded once per cell,
    and single-threaded ONNX makes that the dominant cost. A cell then pays
    only for the chunk texts its configuration actually produces.
    """

    def __init__(self, inner: Embedder) -> None:
        self._inner = inner
        self._memo: dict[str, np.ndarray] = {}
        self.dim = inner.dim
        self.model_id = inner.model_id

    def embed(self, texts: Sequence[str]) -> np.ndarray:
        return self._inner.embed(texts)

    def embed_query(self, text: str) -> np.ndarray:
        if text not in self._memo:
            self._memo[text] = self._inner.embed_query(text)
        return self._memo[text]


def main() -> None:
    """Parse arguments, run every cell, print the table."""
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("dataset", nargs="?", default="datasets/nanorag-docs/dev.jsonl")
    parser.add_argument("--sizes", default="128,256,512,1024")
    parser.add_argument("--overlaps", default="0,32,64")
    parser.add_argument(
        "--models", default="BAAI/bge-base-en-v1.5,BAAI/bge-small-en-v1.5"
    )
    parser.add_argument("--persist-dir", default=".nanorag")
    args = parser.parse_args()

    dataset = load_dataset(args.dataset)
    root = Path(args.persist_dir)
    root.mkdir(parents=True, exist_ok=True)
    print(
        f"dataset {dataset.name}: {len(dataset.items)} items "
        f"({len(dataset.answerable)} answerable), {len(dataset.documents)} documents\n"
    )
    print("| model | chunk | overlap | chunks | " + " | ".join(COLUMNS) + " | secs |")
    rule = " | ".join("---:" for _ in COLUMNS)
    print(f"| --- | ---: | ---: | ---: | {rule} | ---: |")
    for model in args.models.split(","):
        embedder = _QueryMemo(default_embedder(Settings(embedding_model=model), root))
        for size in (int(s) for s in args.sizes.split(",")):
            for overlap in (int(o) for o in args.overlaps.split(",")):
                if overlap >= size:
                    continue
                started = time.perf_counter()
                rag = build_rag(
                    dataset,
                    embedder=embedder,
                    chunker=RecursiveChunker(
                        target_tokens=size, overlap_tokens=overlap
                    ),
                    min_score=default_min_score(model),
                )
                try:
                    report = evaluate_retrieval(dataset, rag)
                finally:
                    rag.close()
                cells = " | ".join(f"{report.metrics[c]:.3f}" for c in COLUMNS)
                print(
                    f"| {model} | {size} | {overlap} | {report.config['chunks']} | "
                    f"{cells} | {time.perf_counter() - started:.0f} |",
                    flush=True,
                )


if __name__ == "__main__":
    main()
