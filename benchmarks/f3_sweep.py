"""MMR and parent-document expansion sweep (plan.md §9 Phase F session F3).

Measures the delta the F3 checkpoint asks for ("Measured delta per
technique"), on the same committed corpus and thresholds every other Phase
D/F sweep uses: the "dense (baseline)" row reproduces `thresholds.json`
exactly, so every MMR/expansion row's delta is directly comparable to it.

Query transforms (`retrieval.query_transform`) are not in this sweep: a
transform needs a real generator call before retrieval even starts, and no
live generator was available this session — the same situation session F1
was in for `JinaReranker`. `LLMQueryTransform`/`MultiQueryRetriever` are
built and unit-tested against fakes; a live measurement is a named
follow-up, not silently skipped.

The corpus is embedded once and reused for every row — only `rag.retriever`
changes between rows (plan.md §5: a plain public attribute), the same
pattern `rerank_sweep.py`/`hybrid_sweep.py` already use.

Run from the repo root::

    uv run python benchmarks/f3_sweep.py
    uv run python benchmarks/f3_sweep.py --candidate-k 32

Prints one Markdown table row per configuration — paste into
`docs/evaluation.md`. Not part of the package and not run by the test suite.
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

from nanorag.config import Settings
from nanorag.evaluation import build_rag, evaluate_retrieval, load_dataset
from nanorag.pipeline import default_embedder, default_min_score
from nanorag.retrieval.dense import DenseRetriever
from nanorag.retrieval.mmr import MmrRetriever
from nanorag.retrieval.parent import ParentExpandingRetriever

COLUMNS = ("recall@1", "recall@5", "recall@10", "mrr", "ndcg@5", "false_abstain_rate")


def main() -> None:
    """Parse args, run every row against one shared embedded index, print rows."""
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("dataset", nargs="?", default="datasets/nanorag-docs/dev.jsonl")
    parser.add_argument("--persist-dir", default=".nanorag")
    parser.add_argument("--candidate-k", type=int, default=32)
    args = parser.parse_args()

    dataset = load_dataset(args.dataset)
    root = Path(args.persist_dir)
    root.mkdir(parents=True, exist_ok=True)
    settings = Settings()
    embedder = default_embedder(settings, root)
    rag = build_rag(
        dataset,
        embedder=embedder,
        min_score=default_min_score(settings.embedding_model),
    )
    print(
        f"dataset {dataset.name}: {len(dataset.items)} items, "
        f"{rag.docs.count_chunks()} chunks\n"
    )
    print("| retriever | param | " + " | ".join(COLUMNS) + " | secs |")
    rule = " | ".join("---:" for _ in COLUMNS)
    print(f"| --- | --- | {rule} | ---: |")

    def _row(name: str, param: str) -> None:
        started = time.perf_counter()
        report = evaluate_retrieval(dataset, rag)
        cells = " | ".join(f"{report.metrics[c]:.3f}" for c in COLUMNS)
        secs = time.perf_counter() - started
        print(f"| {name} | {param} | {cells} | {secs:.0f} |", flush=True)

    dense = DenseRetriever(rag.embedder, rag.vectors, rag.docs, k=rag.retrieve_k)
    dense_min_score = rag.min_score

    try:
        rag.retriever = dense
        _row("dense (baseline)", "-")

        # MMR's ScoredChunk.score is its own composite relevance/diversity
        # signal, not dense cosine (see retrieval.mmr) — the same
        # abstention-scale trap sessions F1/F2 already found for reranker
        # and RRF scores. Off for these rows; there is no measured floor
        # for this scale yet.
        rag.min_score = None
        for lambda_mult in (0.3, 0.5, 0.7):
            rag.retriever = MmrRetriever(
                dense,
                rag.vectors,
                candidate_k=args.candidate_k,
                lambda_mult=lambda_mult,
            )
            _row("mmr", f"lambda={lambda_mult}, candidate_k={args.candidate_k}")

        # ParentExpandingRetriever preserves the wrapped hit's score
        # unchanged (retrieval.parent) — still genuine dense cosine here,
        # so the measured floor applies exactly as it does to the baseline.
        rag.min_score = dense_min_score
        for window in (1, 2):
            rag.retriever = ParentExpandingRetriever(dense, rag.docs, window=window)
            _row("parent-expanding", f"window={window}")
    finally:
        rag.min_score = dense_min_score
        rag.close()


if __name__ == "__main__":
    main()
