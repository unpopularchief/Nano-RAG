"""Search-latency harness — the Phase C acceptance measurement (plan.md §9).

Acceptance: **100k × 768 exact search < 50 ms p95, single-threaded.** This
script builds a ``NumpyVectorStore`` of random unit vectors at that size,
runs unfiltered queries plus two pre-filtered variants (a 50% and a 2%
``allowed_ids`` mask — the shape a metadata filter produces), and prints
p50 / p95 / max per variant. Numbers depend on the machine; the plan's
ceiling is the pass/fail line, not the numbers themselves.

Run from the repo root::

    uv run python benchmarks/search_latency.py            # 100k x 768
    uv run python benchmarks/search_latency.py --rows 20000 --queries 50

Not part of the package and not run by the test suite (the ``-m eval``
harness in Phase D is where committed thresholds live).
"""

from __future__ import annotations

import argparse
import os
import statistics
import time

# Single-threaded, as the acceptance line says. BLAS reads these when numpy
# loads it, so they must be set before the import — not in main().
for _var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_var, "1")

import numpy as np  # noqa: E402

from nanorag.store.numpy_store import NumpyVectorStore  # noqa: E402


def _unit_rows(rng: np.random.Generator, rows: int, dim: int) -> np.ndarray:
    matrix = rng.standard_normal((rows, dim), dtype=np.float32)
    matrix /= np.linalg.norm(matrix, axis=1, keepdims=True)
    return matrix


def _percentiles(samples_ms: list[float]) -> tuple[float, float, float]:
    ordered = sorted(samples_ms)
    p50 = statistics.median(ordered)
    p95 = ordered[min(len(ordered) - 1, int(round(0.95 * (len(ordered) - 1))))]
    return p50, p95, ordered[-1]


def run(
    rows: int, dim: int, queries: int, k: int, seed: int
) -> dict[str, tuple[float, float, float]]:
    """Build the index and time each variant; ``{variant: (p50, p95, max)}`` ms."""
    rng = np.random.default_rng(seed)
    ids = [f"chunk-{i:07d}" for i in range(rows)]
    store = NumpyVectorStore(dim=dim)
    store.upsert(ids, _unit_rows(rng, rows, dim))
    query_vectors = _unit_rows(rng, queries, dim)

    half = set(rng.choice(ids, size=rows // 2, replace=False).tolist())
    two_percent = set(rng.choice(ids, size=max(1, rows // 50), replace=False).tolist())
    variants: dict[str, set[str] | None] = {
        "unfiltered": None,
        "filtered 50%": half,
        "filtered 2%": two_percent,
    }

    results = {}
    for name, allowed in variants.items():
        store.search(query_vectors[0], k, allowed_ids=allowed)  # warm up
        samples = []
        for q in query_vectors:
            started = time.perf_counter()
            store.search(q, k, allowed_ids=allowed)
            samples.append((time.perf_counter() - started) * 1e3)
        results[name] = _percentiles(samples)
    return results


def main() -> None:
    """Parse arguments, run the harness, print a table and the pass/fail line."""
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--rows", type=int, default=100_000)
    parser.add_argument("--dim", type=int, default=768)
    parser.add_argument("--queries", type=int, default=200)
    parser.add_argument("--k", type=int, default=10)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--p95-limit-ms", type=float, default=50.0)
    args = parser.parse_args()

    print(
        f"NumpyVectorStore search: {args.rows} x {args.dim}, k={args.k}, "
        f"{args.queries} queries, single-threaded"
    )
    results = run(args.rows, args.dim, args.queries, args.k, args.seed)
    print(f"{'variant':<14}{'p50 ms':>10}{'p95 ms':>10}{'max ms':>10}")
    for name, (p50, p95, mx) in results.items():
        print(f"{name:<14}{p50:>10.2f}{p95:>10.2f}{mx:>10.2f}")
    p95 = results["unfiltered"][1]
    verdict = "PASS" if p95 < args.p95_limit_ms else "FAIL"
    print(
        f"\nacceptance (unfiltered p95 < {args.p95_limit_ms:.0f} ms): "
        f"{verdict} ({p95:.2f} ms)"
    )


if __name__ == "__main__":
    main()
