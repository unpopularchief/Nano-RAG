"""Retrieval sweep — dense vs. BM25 vs. RRF hybrid (plan.md §9 Phase F session F2).

Measures the delta the F2 checkpoint asks for ("Measured delta vs
dense-only"), on the same committed corpus and thresholds the chunk-size
sweep (``benchmarks/eval_sweep.py``, Phase D3) and the reranking sweep
(``benchmarks/rerank_sweep.py``, session F1) were measured against: the
"dense (baseline)" row here should reproduce ``thresholds.json`` exactly (it
is the same configuration those were measured with), and every BM25/hybrid
row shows its delta from that same baseline.

The corpus is embedded once (the expensive part) and reused for every row —
only ``rag.retriever`` changes between rows, which costs nothing to swap on
an already-built ``Rag`` (plan.md §5: it is a plain public attribute, the
same pattern ``rerank_sweep.py`` already uses for ``rag.reranker``).

Run from the repo root::

    uv run python benchmarks/hybrid_sweep.py
    uv run python benchmarks/hybrid_sweep.py --candidate-ks 8,16,32,64

Prints one Markdown table row per configuration — paste into
``docs/evaluation.md``. Not part of the package and not run by the test
suite.
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

from nanorag.config import Settings
from nanorag.evaluation import build_rag, evaluate_retrieval, load_dataset
from nanorag.pipeline import default_embedder, default_min_score
from nanorag.retrieval.bm25 import Bm25Retriever
from nanorag.retrieval.dense import DenseRetriever
from nanorag.retrieval.hybrid import HybridRetriever

COLUMNS = (
    "recall@1",
    "recall@5",
    "recall@10",
    "mrr",
    "ndcg@5",
    "false_abstain_rate",
    "abstain_rate",
)


def main() -> None:
    """Parse args, run every row against one shared embedded index, print rows."""
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("dataset", nargs="?", default="datasets/nanorag-docs/dev.jsonl")
    parser.add_argument("--persist-dir", default=".nanorag")
    parser.add_argument("--candidate-ks", default="8,16,32")
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
        f"{rag.docs.count_chunks()} chunks, "
        f"fts5_available={rag.docs.fts5_available}\n"
    )
    print("| retriever | candidate_k | " + " | ".join(COLUMNS) + " | secs |")
    rule = " | ".join("---:" for _ in COLUMNS)
    print(f"| --- | ---: | {rule} | ---: |")

    def _row(name: str, candidate_k: int) -> None:
        started = time.perf_counter()
        report = evaluate_retrieval(dataset, rag)
        cells = " | ".join(f"{report.metrics[c]:.3f}" for c in COLUMNS)
        secs = time.perf_counter() - started
        print(f"| {name} | {candidate_k} | {cells} | {secs:.0f} |", flush=True)

    dense = DenseRetriever(rag.embedder, rag.vectors, rag.docs, k=rag.retrieve_k)
    bm25 = Bm25Retriever(rag.docs, k=rag.retrieve_k)
    dense_min_score = rag.min_score

    try:
        rag.retriever = dense
        _row("dense (baseline)", rag.retrieve_k)

        # `min_score` is calibrated against the *dense* embedder's cosine
        # scale (plan.md §18 F10) — BM25's and RRF's scores are on their own,
        # much smaller scales (RRF: ~0.01-0.05; BM25: unbounded but usually
        # nowhere near cosine's 0-1 range), so the dense floor makes every
        # one of them read as "insufficient" regardless of relevance. A
        # first run of this sweep with the floor left on measured
        # `false_abstain_rate = 1.000` for every hybrid row — the same
        # abstention-scale trap session F1 found for reranker scores,
        # confirmed here rather than assumed. Off entirely: neither scale
        # has a measured floor of its own yet.
        rag.min_score = None

        rag.retriever = bm25
        _row("bm25", rag.retrieve_k)

        for candidate_k in (int(x) for x in args.candidate_ks.split(",")):
            rag.retriever = HybridRetriever(dense, bm25, candidate_k=candidate_k)
            _row("hybrid (rrf)", candidate_k)
    finally:
        rag.min_score = dense_min_score
        rag.close()


if __name__ == "__main__":
    main()
