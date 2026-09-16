"""Reranking sweep — identity vs the local ONNX cross-encoder (plan.md §9 Phase F).

Measures the nDCG@5 delta the F1 checkpoint asks for, on the same committed
corpus and thresholds the chunk-size sweep (``benchmarks/eval_sweep.py``,
Phase D3) was measured against, so the numbers are directly comparable: the
identity row here should reproduce ``thresholds.json`` (it is the same
configuration pre-Phase-F retrieval always used), and every cross-encoder
row shows its delta from that same baseline.

The corpus is embedded once (the expensive part, ~10 min cold with the real
default embedder) and reused for every row — only the reranker and
``retrieve_k`` change between rows, which costs nothing to swap on an
already-built ``Rag`` (plan.md §5: it is a plain public attribute).

Run from the repo root::

    uv run python benchmarks/rerank_sweep.py
    uv run python benchmarks/rerank_sweep.py --retrieve-ks 8,16,32,64

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
from nanorag.rerank.identity import IdentityReranker
from nanorag.rerank.local_cross_encoder import LocalCrossEncoderReranker

COLUMNS = ("recall@1", "recall@5", "recall@10", "mrr", "ndcg@5", "false_abstain_rate")


def main() -> None:
    """Parse args, run every row against one shared embedded index, print rows."""
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("dataset", nargs="?", default="datasets/nanorag-docs/dev.jsonl")
    parser.add_argument("--persist-dir", default=".nanorag")
    parser.add_argument("--retrieve-ks", default="8,16,32")
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
    print("| reranker | retrieve_k | " + " | ".join(COLUMNS) + " | secs |")
    rule = " | ".join("---:" for _ in COLUMNS)
    print(f"| --- | ---: | {rule} | ---: |")

    def _row(name: str, retrieve_k: int) -> None:
        started = time.perf_counter()
        report = evaluate_retrieval(dataset, rag)
        cells = " | ".join(f"{report.metrics[c]:.3f}" for c in COLUMNS)
        secs = time.perf_counter() - started
        print(f"| {name} | {retrieve_k} | {cells} | {secs:.0f} |", flush=True)

    try:
        rag.reranker = IdentityReranker()
        rag.retrieve_k = rag.k
        _row("identity (baseline)", rag.retrieve_k)

        cross_encoder = LocalCrossEncoderReranker()
        for retrieve_k in (int(x) for x in args.retrieve_ks.split(",")):
            rag.reranker = cross_encoder
            rag.retrieve_k = retrieve_k
            _row("local cross-encoder", retrieve_k)
    finally:
        rag.close()


if __name__ == "__main__":
    main()
