# Evaluation

Retrieval quality is a number in CI, not a vibe (plan.md §1). This page
is the record: what is measured, on what, how the baseline was taken, what
the sweep showed, and the numbers the gate holds the pipeline to.

## What is measured

Two runs, both in `nanorag/evaluation/` and both behind `nanorag eval`:

- **Retrieval** (`evaluate_retrieval`) — offline, keyless. The frozen
  corpus is ingested through the real write path into an in-memory index,
  every question is retrieved through the real read path, and the ranking
  is graded against gold spans. This is the run that gates CI.
- **Answers** (`evaluate_answers`) — one generation call per item. Grades
  abstention, citations and a lexical answer signal. Manual, never in CI;
  its job is the two-generator comparison plan.md §9 D3 asks for, so a
  prompt tuned on a large API model is not shipped blind to small local
  ones.

The dataset is [`datasets/nanorag-docs/`](../datasets/nanorag-docs/README.md):
19 documents (this project's own docs plus twelve permissively licensed
package READMEs, frozen), **341 items — 317 answerable, 24 unanswerable**.
Gold is a verbatim quote resolved to character offsets in its document, so
a retrieved chunk is *relevant* when it overlaps a gold span and the same
dataset grades every chunker, chunk size and overlap alike.

## Metrics

Retrieval, per answerable item, averaged over the 317 (`k` counts
positions; a position past the end of the list is irrelevant):

| Metric | Definition |
| --- | --- |
| `recall@k` | gold spans covered by at least one of the top *k* chunks ÷ gold spans |
| `precision@k` | top-*k* positions that overlap a gold span ÷ *k* |
| `hit_rate@k` | 1 if any of the top *k* is relevant |
| `mrr` | 1 ÷ rank of the first relevant chunk (0 if none) |
| `ndcg@k` | DCG ÷ ideal DCG with binary gains, where a position gains only if it covers a gold span **no earlier position covered** — with overlapping chunks two positions often cover one span, and counting both would push nDCG past 1 |

Abstention, from the same run under the pipeline's `insufficient_context`
thresholds: `false_abstain_rate` (answerable items flagged) and
`abstain_rate` (unanswerable items flagged). Answer runs report
`answer_rate` (answerable items not abstained on), `abstain_rate`
(unanswerable items abstained on), `cited_rate`, `citation_validity`
(markers that resolved), `citation_precision` (resolved citations that
point at a gold span — the model cited *where the answer is*, not merely
something in the prompt) and `token_f1` (SQuAD-style bag-of-tokens F1
against the item's short reference answer, where one exists). A citation
proves provenance, not truth, and no metric here claims otherwise.

Every metric is a pure function in `evaluation/retrieval_metrics.py` /
`answer_metrics.py`, tested against hand-computed values on toy rankings
(ties, empty gold sets, positions past the list) and by property.

## Running it

```bash
uv sync --extra dev --extra local
uv run nanorag eval datasets/nanorag-docs/dev.jsonl                      # retrieval, text report
uv run nanorag eval datasets/nanorag-docs/dev.jsonl --json > report.json # every item row
uv run nanorag eval datasets/nanorag-docs/dev.jsonl \
    --thresholds datasets/nanorag-docs/thresholds.json                   # exit 6 on a regression
uv run nanorag eval datasets/nanorag-docs/dev.jsonl --chunk-tokens 256 --overlap-tokens 32
uv run nanorag eval datasets/nanorag-docs/dev.jsonl --answers --generator groq
uv run pytest -q -m eval -s                                              # the CI gate, locally
uv run python benchmarks/eval_sweep.py                                   # the sweep below
```

The corpus is embedded once per configuration and cached under
`--persist-dir` — the slow part, five or six minutes on a laptop at
single-threaded ONNX for the 128-token default, seconds once cached;
questions are embedded on every run (queries are never cached, by design),
under a minute for the 341. `.github/workflows/eval.yml` runs the gate nightly and on
demand on one fixed runner image (`ubuntu-24.04`), not the CI matrix: ONNX
Runtime is not bit-identical across builds and CPUs, and the tolerance
band is what absorbs that drift (plan.md §18 F3, F5).

## The sweep — chunk size × overlap × embedding model

`benchmarks/eval_sweep.py`, 2026-09-16, dev laptop, single-threaded ONNX,
`RecursiveChunker` under its own heuristic counter (so "128 tokens" is
about 512 characters), retrieval graded at `k = 1, 3, 5, 10`
(`mrr` and `ndcg` are computed over the top 10). Every cell is the same
341 questions against the same 19 documents; only the chunker and the
embedder change.

| model | chunk | overlap | chunks | recall@1 | recall@5 | recall@10 | mrr | ndcg@5 | false_abstain |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| bge-base-en-v1.5 | 128 | 0 | 520 | 0.503 | 0.768 | 0.842 | 0.620 | 0.649 | 0.006 |
| bge-base-en-v1.5 | 128 | 32 | 598 | 0.530 | 0.806 | 0.864 | 0.641 | 0.676 | 0.003 |
| **bge-base-en-v1.5** | **128** | **64** | **839** | **0.544** | **0.836** | **0.880** | **0.659** | **0.698** | **0.003** |
| bge-base-en-v1.5 | 256 | 0 | 256 | 0.451 | 0.754 | 0.833 | 0.581 | 0.617 | 0.006 |
| bge-base-en-v1.5 | 256 | 32 | 272 | 0.442 | 0.749 | 0.849 | 0.567 | 0.602 | 0.003 |
| bge-base-en-v1.5 | 256 | 64 | 305 | 0.451 | 0.754 | 0.858 | 0.586 | 0.617 | 0.003 |
| bge-base-en-v1.5 | 512 | 0 | 126 | 0.431 | 0.700 | 0.808 | 0.546 | 0.574 | 0.006 |
| bge-base-en-v1.5 | 512 | 32 | 129 | 0.423 | 0.719 | 0.820 | 0.548 | 0.581 | 0.003 |
| bge-base-en-v1.5 | 512 | 64 *(old default)* | 136 | 0.407 | 0.694 | 0.801 | 0.529 | 0.558 | 0.003 |
| bge-base-en-v1.5 | 1024 | 0 | 65 | 0.416 | 0.644 | 0.760 | 0.513 | 0.533 | 0.006 |
| bge-base-en-v1.5 | 1024 | 32 | 67 | 0.401 | 0.640 | 0.773 | 0.510 | 0.529 | 0.003 |
| bge-base-en-v1.5 | 1024 | 64 | 70 | 0.394 | 0.659 | 0.757 | 0.509 | 0.536 | 0.003 |
| bge-small-en-v1.5 | 128 | 0 | 520 | 0.483 | 0.763 | 0.845 | 0.593 | 0.627 | — |
| bge-small-en-v1.5 | 128 | 32 | 598 | 0.521 | 0.754 | 0.842 | 0.622 | 0.646 | — |
| bge-small-en-v1.5 | 128 | 64 | 839 | 0.544 | 0.785 | 0.864 | 0.647 | 0.673 | — |
| bge-small-en-v1.5 | 256 | 0 | 256 | 0.461 | 0.744 | 0.830 | 0.585 | 0.616 | — |
| bge-small-en-v1.5 | 256 | 32 | 272 | 0.470 | 0.751 | 0.842 | 0.588 | 0.619 | — |
| bge-small-en-v1.5 | 256 | 64 | 305 | 0.435 | 0.767 | 0.845 | 0.574 | 0.613 | — |
| bge-small-en-v1.5 | 512 | 0 | 126 | 0.431 | 0.729 | 0.833 | 0.560 | 0.592 | — |
| bge-small-en-v1.5 | 512 | 32 | 129 | 0.402 | 0.713 | 0.811 | 0.537 | 0.571 | — |
| bge-small-en-v1.5 | 512 | 64 | 136 | 0.420 | 0.707 | 0.820 | 0.547 | 0.575 | — |
| bge-small-en-v1.5 | 1024 | 0 | 65 | 0.347 | 0.688 | 0.789 | 0.485 | 0.524 | — |
| bge-small-en-v1.5 | 1024 | 32 | 67 | 0.347 | 0.669 | 0.795 | 0.488 | 0.519 | — |
| bge-small-en-v1.5 | 1024 | 64 | 70 | 0.372 | 0.669 | 0.779 | 0.497 | 0.528 | — |

(`false_abstain` is `—` for bge-small because no abstention floor is
measured for it; `Rag` applies none, and the column would read 0 by
construction.)

What it says:

1. **Chunk size dominates, and smaller is better at every cut-off.**
   Recall@5 with bge-base runs 0.77–0.84 at 128 tokens, 0.75 at 256,
   0.69–0.72 at 512 and 0.64–0.66 at 1024 — a 14-pp spread, three times
   the tolerance band. Gold here is a sentence or a short paragraph; a
   512-token chunk carries the answer plus three or four unrelated
   paragraphs that dilute its vector.
2. **Overlap matters only at 128 tokens**, where 50 % overlap (64) is
   worth +7 pp Recall@5 over none — at that size a boundary cuts through
   an answer often enough that the second copy pays for itself. At 256 and
   above the three overlaps are within 2 pp of each other, i.e. within
   noise.
3. **bge-small is within 1–5 pp of bge-base everywhere** and marginally
   ahead at 512/1024. bge-base stays the default: it wins at the chosen
   chunk size (Recall@5 0.836 vs 0.785, MRR 0.659 vs 0.647) and is the only
   model with a measured abstention floor. bge-small is a sound choice
   where the 440 MB download or CPU time matters; the numbers above are its
   baseline.

The fixed-`k` comparison flatters small chunks — five 128-token chunks are
a quarter of the text of five 512-token ones. Holding the **token budget**
roughly constant instead (`--ks 1,3,5,8,10,20`, bge-base, overlap 64):

| chunk | ≈ 1k tokens | ≈ 2–2.5k tokens | ≈ 5k tokens | ≈ 10k tokens |
| ---: | ---: | ---: | ---: | ---: |
| 128 | recall@8 **0.874** | recall@20 **0.924** | | |
| 256 | recall@5 0.754 | recall@8 0.836, @10 0.858 | recall@20 0.899 | |
| 512 | | recall@5 0.694 | recall@8 0.763, @10 0.801 | recall@20 0.902 |

Same spend, more of the gold: at ~2.5k tokens of context, 128-token chunks
retrieve 92 % of the gold spans against 69 % for the old default. The
sweep's conclusion holds under either accounting.

## Answers — the two-generator comparison

`nanorag eval --answers`, `k = 8` unless stated, the same 341 items. The
local leg is complete; the free-tier API legs are not, and that is a
finding in itself (below).

| generator | chunk | k | completed | answer_rate | abstain_rate | cited_rate | citation_validity | citation_precision | token_f1 | mean tokens | mean ms |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Ollama `qwen2.5:7b-instruct` | 512/64 | 8 | 1.00 | 0.785 | 1.000 | 0.992 | 0.996 | 0.609 | 0.200 | 2588 † | 1455 |
| Ollama `qwen2.5:7b-instruct` | 256/64 | 8 | 1.00 | 0.864 | 1.000 | 1.000 | 0.998 | 0.621 | 0.242 | 2382 | 1328 |
| **Ollama `qwen2.5:7b-instruct`** | **128/64** | **8** | 1.00 | **0.918** | 0.958 | 1.000 | 0.999 | **0.690** | **0.266** | **1418** | **993** |
| Ollama `qwen2.5:7b-instruct` | 128/64 | 16 | 1.00 | 0.943 | 0.958 | 1.000 | 1.000 | 0.621 | 0.259 | 2430 | 1386 |
| Gemini `gemini-3.6-flash` | 512/64 | 8 | 0.04 | 0.929 | — | 1.000 | 1.000 | 0.500 | 0.282 | 5087 | 10084 |
| Groq `openai/gpt-oss-120b` | 512/64 | 8 | partial | *(no report — see below)* | | | | | | | |

† every query truncated: with Ollama's 4096-token window only ~4 of the
8 chunks fit at 512 tokens.

Reading the local leg: the answer rate — answerable questions the model
actually answered rather than abstained on — climbs from 79 % to 92 % as
chunks shrink, citation precision (citations that point at a gold span)
from 0.61 to 0.69, and the lexical F1 against the reference answers
from 0.20 to 0.27, while the prompt halves. Part of the 512 → 256 step is
the window (at 512 the prompt was cut on every query); the 256 → 128 step
is not — nothing was truncated at either size — so it is retrieval
quality reaching the model. `k = 16` at 128 tokens buys another 2.5 pp of
answer rate for 7 pp of citation precision and 70 % more tokens: the
model cites more of the extra, mostly irrelevant chunks (precision@16 is
under 0.1). That trade is a caller's `k=` to make, not the default's.

**The free-tier API legs did not complete.** Groq's `openai/gpt-oss-120b`
answered part of the set (about half an hour of calls) and then returned a
daily-cap 429 (`QuotaExhausted` — tokens per day: each prompt is ~2.5k
tokens, so the full set is ~850k, well past the free tier's daily token
allowance). Gemini `gemini-3.6-flash`
answered 14 and then rate-limited every remaining item. plan.md §4 sized
the free tiers for "the handful of queries per day you actually run"; a
341-item answer eval is not that, and it should not be run as one. Two
things came out of this:

- `evaluate_answers` no longer loses a run to a failure at item 80: a
  provider error is recorded on the item and the run continues, a
  `QuotaExhausted` / `AuthError` or ten consecutive failures stop it, and
  `completed_rate` says how much of the set the other numbers cover (the
  Gemini row above is what that looks like; the Groq run predates it and
  saved nothing).
- The API-model leg is a Gate D item: run it on a **stratified subset**
  (a `--limit`/`--tags` selection is a small addition) or spread over
  days, and compare 128/64 against 512/64 the way the local leg does. The
  14 Gemini items that did complete (all answerable; 13 answered, 1
  abstained, every answer cited) are too few to read beyond "the large
  model at 512/64 was already about where the 7B model gets to at
  128/64" — the usual shape (plan.md §11: prompt weaknesses vs model
  weaknesses), and exactly why the small-model leg is the one that
  decides the default.

## Baseline and thresholds

**Chunker default changed by this measurement:** `RecursiveChunker` (and
`FixedChunker`, `MarkdownChunker`) now default to `target_tokens=128,
overlap_tokens=64`, from 512 / 64. `Rag()` and `from_defaults()` inherit
it. The gate baseline is that configuration, measured on 2026-09-16 on the
dev laptop and reproduced to four decimals by `pytest -m eval`:

| metric | baseline | tolerance | floor |
| --- | ---: | ---: | ---: |
| `recall@5` | 0.836 | 0.05 | 0.786 |
| `recall@10` | 0.880 | 0.05 | 0.830 |
| `mrr` | 0.659 | 0.05 | 0.609 |
| `ndcg@5` | 0.698 | 0.05 | 0.648 |

with `precision@5` 0.216, `hit_rate@1` 0.546, `false_abstain_rate` 0.003
(1 of 317 answerable items flagged under the 0.55 floor) and
`abstain_rate` 0.417 — 10 of the 24 unanswerable items flagged: 10 of the
12 off-topic ones (the two misses are "change a flat tyre on a bicycle" at
0.57 and the gibberish string at 0.61 — the gibberish-scores-high trap
already seen at Gate C) and none of the 12 on-topic-but-uncovered ones,
which score 0.67–0.87 against 128-token chunks. Those are the model's to
abstain on, and the local model did on 11 of the 12 (it answered "how
does the Qdrant store implement metadata filters" from the filter-grammar
docs — a plausible-sounding conflation, and the kind of miss the flag
cannot catch).

**Why 0.05.** The gate must not trip on noise (plan.md §18 F3). The noise
that matters is sampling: the standard error of a proportion at
n = 317 is √(p(1−p)/317) ≈ 0.021 at p = 0.84 and 0.018 at p = 0.88, so
0.05 is roughly two standard errors — a drop that size is unlikely to be
chance and, at 5 pp on 317 items, is 16 questions that stopped
retrieving. Build-to-build ONNX drift is far smaller: it moves scores in
the fourth decimal and can flip a near-tie, which is a fraction of a
point on these averages. F3's alternative of a "large regressions only"
gate with a ≥ 6 pp margin was for a ~50-item set; at 317 items the
tighter band is justified. The four gated metrics are the ones the
default configuration is chosen on; `precision@k` and the abstention
rates are reported, not gated (precision is bounded by how many chunks
overlap one span, and the abstention floor is a separate, Gate C
decision).

**Re-baselining.** A deliberate change that moves the numbers — a new
default chunker, embedder, or retriever — re-runs `nanorag eval
datasets/nanorag-docs/dev.jsonl` on the CI runner image and copies the
metrics into `thresholds.json` in the same PR, with the old and new
numbers in `CHANGELOG.md`. A floor is never lowered to make a red run
green, and the corpus and items are never edited to fit a result: a
changed corpus is a new dataset.

## Reranking (Phase F, session F1)

**Measured with `benchmarks/rerank_sweep.py`**, on the same committed
corpus, thresholds and 128/64 chunker default as above — the identity row
reproduces the baseline exactly, so every cross-encoder row's delta is
directly comparable to it, not a second, differently-configured baseline.

| reranker | retrieve_k | recall@1 | recall@5 | recall@10 | mrr | ndcg@5 | false_abstain | secs |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| identity (baseline) | — | 0.544 | 0.836 | 0.880 | 0.659 | 0.698 | 0.003 | 76 |
| local cross-encoder | 8 | 0.699 | 0.861 | 0.880 | 0.771 | 0.792 | 0.003 | 117 |
| local cross-encoder | 16 | 0.705 | 0.880 | 0.899 | 0.780 | 0.803 | 0.003 | 122 |
| local cross-encoder | 32 | 0.721 | 0.912 | 0.937 | 0.802 | 0.827 | 0.003 | 145 |
| local cross-encoder | 64 | 0.721 | 0.915 | 0.959 | 0.808 | 0.830 | 0.003 | 240 |

**`LocalCrossEncoderReranker` (`Xenova/ms-marco-MiniLM-L-6-v2`, offline)
beats the Phase D baseline by a wide, unambiguous margin** — `ndcg@5`
+0.129 (0.698 → 0.827) at `retrieve_k=32`, roughly **2.6× the 0.05
tolerance band**, with every other gated metric improving alongside it.
`recall@1` almost jumps a third (0.544 → 0.721): the cross-encoder scores
each candidate against the actual question text rather than a single
cosine number, and clearly resolves ties and near-misses dense retrieval
cannot. **Gains grow with `retrieve_k` but visibly plateau**: 8 → 16 adds
+0.011 `ndcg@5`, 16 → 32 adds +0.024, 32 → 64 adds only +0.003 — for
roughly 1.7× the reranking time (145 s → 240 s over 341 items). **32 is
the picked oversampling factor**: it captures effectively all of the
measured gain (0.827 of 0.830) at meaningfully lower cost than 64.

**The abstention floor must be judged on the retriever's own dense
scores, never the reranker's** — a real bug this sweep caught, not a
theoretical concern. The first sweep run (before the fix) showed
`false_abstain_rate` jump from 0.003 to 0.10–0.13 with the cross-encoder
active: `insufficient_context()`'s `min_score`/`min_gap` floors are
calibrated against the *embedder's* cosine scale (plan.md §18 F10,
`docs/conventions.md` "Abstention"), and a cross-encoder's relevance
score is a different, uncalibrated scale entirely — reusing it for the
same floor made a third of genuinely on-topic queries look thin. Fixed in
`Rag.query()` and `evaluate_retrieval()` (both now call
`rag.retriever.retrieve()` directly for the abstention check, and
`pipeline.rerank_hits()` — a new module-level function alongside
`context_budget`/`insufficient_context`, plan.md §5 rule 1 — separately
for the hits that actually go in the prompt): confirmed by the corrected
sweep above, where every reranked row's `false_abstain_rate` matches the
baseline's 0.003 exactly.

**Not wired as `from_defaults()`'s default, despite clearing the
Acceptance bar, for a reason the sweep didn't measure: model-load
cost, not retrieval quality.** `LocalCrossEncoderReranker.__init__`
eagerly loads its ONNX model, the same way `FastEmbedEmbedder` does — but
`from_defaults()` builds one `Rag` for every CLI command, including
`nanorag ingest`, which plan.md §4 states plainly needs no network at
all. Defaulting the reranker on would make a bare `nanorag ingest` load
(and, on a cold cache, download) a second ~80 MB model it will never use.
Fixing this properly needs lazy reranker construction — the model loads
on first `retrieve()`/`query()` call, not at `Rag()`/`from_defaults()`
construction time — which is real, scoped-out follow-up work, not
something to bundle silently into F1. **Until then, wire it in by hand**
for the query path specifically:

```python
from nanorag.rerank import LocalCrossEncoderReranker

rag = Rag.from_defaults(reranker=LocalCrossEncoderReranker(), retrieve_k=32)
```

`Rag()`'s own bare constructor default stays `IdentityReranker` regardless
— a caller who wires every component by hand and never touches
reranking must not suddenly need the `[local]` extra just because they
constructed a `Rag`.

**`JinaReranker` was built and unit-tested (via `MockTransport`) but not
measured live** — no `JINA_API_KEY` was available this session, matching
Gate D's Groq/Gemini situation. Its error mapping is inferred from the
general Jina API shape, documented as such in `nanorag/rerank/jina.py`.
Running the same head-to-head sweep against it, and comparing to the
local cross-encoder within noise (plan.md §9 Phase F Acceptance: "the
local cross-encoder is preferred... if it matches within noise"), is a
named follow-up, not a Gate F blocker — the local cross-encoder's own
result already stands on its own measured merit.

## Hybrid retrieval (Phase F, session F2)

**Measured with `benchmarks/hybrid_sweep.py`**, on the same committed
corpus, thresholds and 128/64 chunker default as above — the "dense
(baseline)" row reproduces `thresholds.json` exactly, so every BM25/hybrid
row's delta is directly comparable to it.

| retriever | candidate_k | recall@1 | recall@5 | recall@10 | mrr | ndcg@5 | false_abstain | abstain_rate | secs |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| dense (baseline) | — | 0.544 | 0.836 | 0.880 | 0.659 | 0.698 | 0.003 | 0.417 | 59 |
| bm25 | — | 0.632 | 0.877 | 0.921 | 0.734 | 0.764 | 0.000 | 0.042 | 0 |
| hybrid (rrf) | 8 | 0.658 | 0.896 | 0.968 | 0.764 | 0.790 | 0.000 | 0.000 | 59 |
| hybrid (rrf) | 32 | 0.670 | 0.893 | 0.953 | 0.770 | 0.795 | 0.000 | 0.000 | 57 |
| hybrid (rrf) | 64 | 0.670 | 0.890 | 0.959 | 0.770 | 0.793 | 0.000 | 0.000 | 57 |

**BM25 alone beats the Phase D dense baseline by a wide margin on this
corpus** — `ndcg@5` +0.066 (0.698 → 0.764), `recall@1` +0.088 (0.544 →
0.632) — essentially free (SQLite FTS5 over 839 chunks scores in well under
a second, the "0" in the `secs` column). This corpus is technical
documentation — flag names, function names, exact filenames — exactly the
vocabulary lexical search is strong at and where a paraphrase-tolerant
embedding sometimes blurs a specific term into its neighbourhood instead of
matching it exactly. **RRF hybrid fusion beats both single-signal
retrievers on every ranking metric**, `ndcg@5` reaching 0.795 at
`candidate_k=32` (+0.097 over dense, +0.031 over BM25 alone) — the two
signals catch different chunks often enough that combining them, even by
rank alone, finds more of the gold spans than either does. Gains from
widening `candidate_k` past 32 are flat (`ndcg@5` 0.795 → 0.793, within
noise), mirroring session F1's reranking sweep's own plateau shape.

**A second abstention-scale bug, the same shape as F1's, found by running
the real measurement rather than assumed:** the first run of this sweep,
with `Rag`'s dense-calibrated `min_score` (:data:`DEFAULT_MIN_SCORE`,
`0.55`) left untouched, measured `false_abstain_rate = 1.000` for every
hybrid row — literally every answerable question flagged as insufficient.
Cause: RRF's fused scores sit around `0.01`–`0.05` (a handful of
`1 / (60 + rank)` terms), and BM25's are on their own unbounded-but-usually-
larger scale — neither is remotely close to a cosine floor measured for the
*embedder*. `docs/conventions.md` "Abstention" and `pipeline.py`'s
docstrings now say plainly that `min_score` is meaningful only for
whichever retriever is actually wired into `self.retriever`, generalising
the caveat F1 already established for reranker scores. The sweep script
sets `rag.min_score = None` before measuring the BM25/hybrid rows, which is
what the table above reports.

**That fix surfaces the real, unresolved cost of switching retrievers: with
no calibrated floor, `insufficient_context` stops catching genuinely
unrelated questions.** `abstain_rate` (the share of unanswerable items
correctly flagged) drops from the dense baseline's measured `0.417` to
`0.042` for BM25 and `0.000` for hybrid — BM25 and RRF scores for an
off-topic query are not reliably lower than for an on-topic one the way
dense cosine is, so a `min_score` floor for these scales, if one exists at
all, has not been measured. **This is the reason `Bm25Retriever` /
`HybridRetriever` are not wired into `Rag.from_defaults()`'s default**,
despite clearing the Acceptance bar on ranking metrics alone: unlike F1's
reranker (a model-loading cost, unrelated to retrieval quality), this is a
genuine, currently-unmitigated quality regression on the "say I don't know"
half of the engine. Measuring a BM25/RRF-scale abstention floor — the same
kind of study `insufficient_context`'s docstring already did for the
default dense embedder — is a named follow-up, not silently dropped.

**FTS5 is a capability check, not an assumption, and both paths are
measured to rank the same way.** `SqliteDocumentStore` tries to create the
`chunks_fts` virtual table at open time and records whether it worked as
`fts5_available` — `True` on every machine this session ran on (Python's
bundled SQLite ships FTS5 by default on both Windows and the `ubuntu-24.04`
CI image), but `tests/test_bm25_retriever.py` and
`tests/test_sqlite_docs.py` force the unavailable branch directly (there is
no portable way to actually build a Python without FTS5 compiled in to test
against) and confirm `Bm25Retriever`'s NumPy fallback — the same Okapi
BM25 formula (`k1=1.2`, `b=0.75`, SQLite's own `bm25()` defaults) computed
from scratch per query instead of from an index — ranks the toy corpus
identically to the FTS5 path. **Zero new dependencies either way**: FTS5
ships inside SQLite itself, and the fallback is stdlib `re` plus the
already-core `numpy` — `uv.lock` is unchanged by this session.

**Wire either in by hand** (both stay opt-in; `Rag()`'s and
`from_defaults()`'s own default remains `DenseRetriever`):

```python
from nanorag.retrieval import Bm25Retriever, DenseRetriever, HybridRetriever

rag = Rag.from_defaults(min_score=None)  # no measured floor for these scales yet
dense = DenseRetriever(rag.embedder, rag.vectors, rag.docs, k=rag.retrieve_k)
bm25 = Bm25Retriever(rag.docs, k=rag.retrieve_k)
rag.retriever = bm25  # BM25 alone, or:
rag.retriever = HybridRetriever(dense, bm25, candidate_k=32)  # the best-measured row above
```

## MMR, parent-document expansion, query transforms (Phase F, session F3)

**Measured with `benchmarks/f3_sweep.py`**, on the same committed corpus,
thresholds and 128/64 chunker default as above — the "dense (baseline)"
row reproduces `thresholds.json` exactly.

| retriever | param | recall@1 | recall@5 | recall@10 | mrr | ndcg@5 | false_abstain | secs |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| dense (baseline) | — | 0.544 | 0.836 | 0.880 | 0.659 | 0.698 | 0.003 | 59 |
| mmr | lambda=0.3, candidate_k=32 | 0.544 | 0.823 | 0.886 | 0.655 | 0.690 | 0.000 | 57 |
| mmr | lambda=0.5, candidate_k=32 | 0.544 | 0.836 | 0.890 | 0.659 | 0.697 | 0.000 | 57 |
| mmr | lambda=0.7, candidate_k=32 | 0.544 | 0.839 | 0.880 | 0.661 | 0.701 | 0.000 | 58 |
| parent-expanding | window=1 | 0.569 | 0.861 | 0.909 | 0.684 | 0.723 | 0.003 | 58 |
| parent-expanding | window=2 | 0.582 | 0.874 | 0.924 | 0.696 | 0.735 | 0.003 | 55 |

**MMR is a genuine negative result on this corpus — the honest kind Phase
F's own Acceptance criterion asks for ("every delta, including negative
ones, documented"), the first Phase F technique that does not clear the
bar.** Every `lambda_mult` tested lands within noise of the baseline
(`ndcg@5` 0.690–0.701 vs. 0.698) — diversifying away from the single best
match neither helps nor meaningfully hurts, because this corpus rarely
puts several near-duplicate chunks in one query's top candidates to begin
with (`nanorag-docs` is technical documentation with one authoritative
place for most facts, not a corpus with redundant restatements across
files). MMR's mechanism has nothing to trade against on a query like that:
trading a small amount of relevance for diversity when there is no
redundant runner-up to diversify *away from* just removes a good hit.
**Not enabled by default** — correctly, since it does not beat the
baseline — and left available for corpora where genuine near-duplication
in the top candidates is a real problem (a corpus scraped from several
overlapping sources, say).

**Parent-document (neighbour-window) expansion is a real, if modest, win
— `ndcg@5` +0.025 (window=1) to +0.037 (window=2) — but it measures a
different thing than F1's/F2's wins, worth being precise about.**
`ParentExpandingRetriever` never changes *which* chunk ranks where (its
docstring states this as a guarantee, and the sweep's ranking order is
identical to the dense baseline's own) — it only widens each hit's
`start_char`/`end_char` before ``recall``/``ndcg`` check whether that span
overlaps a gold quote. A narrow chunk whose boundary just misses the gold
span often has that gold span fall inside its *widened* neighbour window
instead, which is exactly why `recall@1` moves (0.544 → 0.582) even though
the top-ranked hit is the same chunk before and after expansion. This is a
real effect with genuine production value (a wider excerpt is more likely
to actually contain the sentence a citation needs, which is the entire
point of small-to-big retrieval) — but it is a measurement of "does the
shown span cover the answer," not "did retrieval rank the right document
higher," and the two should not be conflated when reading the numbers.
**Not enabled by default**, for two reasons named in `retrieval/parent.py`
rather than discovered here: (1) a wider chunk consumes more of the
context token budget per hit, so `ContextBuilder` fits fewer hits before
truncating — a real trade-off this retrieval-only sweep cannot see, since
it never runs the full `query()` path with a real budget; (2) two hits
adjacent in the same result list expand into overlapping-but-not-identical
spans that `ContextBuilder`'s exact-text dedup does not catch, wasting
some of that budget on repeated sentences. Both are documented limitations
in the module, not fixed here. A full `evaluate_answers` run (answer
quality, not just retrieval-span overlap) with expansion wired in is a
named follow-up.

**Query transforms (`retrieval.query_transform`) were built and
unit-tested against fakes but not measured live — no generator was
available this session, the same situation F1's `JinaReranker` was in.**
`LLMQueryTransform` needs a real model call *before* retrieval even
starts — the one deliberate exception to plan.md §4's "the only network
call in the whole engine is the single generation request per query" —
which is exactly why `QueryTransform` is opt-in behind an explicit
constructor argument and `IdentityQueryTransform` (a true no-op — see
`retrieval/query_transform.py`) is `MultiQueryRetriever`'s default. A live
sweep comparing `MultiQueryRetriever` against a plain retriever, once a
generator is available, is a named follow-up, not silently dropped.

**Wire any of these in by hand** (all three stay opt-in):

```python
from nanorag.retrieval import DenseRetriever, MmrRetriever, MultiQueryRetriever
from nanorag.retrieval.parent import ParentExpandingRetriever
from nanorag.retrieval.query_transform import LLMQueryTransform

dense = DenseRetriever(rag.embedder, rag.vectors, rag.docs, k=rag.retrieve_k)
rag.retriever = MmrRetriever(dense, rag.vectors, candidate_k=32, lambda_mult=0.7)
# or, the best-measured configuration above:
rag.retriever = ParentExpandingRetriever(dense, rag.docs, window=2)
# or, once a generator is available for the extra call:
rag.retriever = MultiQueryRetriever(dense, LLMQueryTransform(rag.generator))
```

## Negative results and caveats

- **Overlap at 256 tokens and above buys nothing measurable** (≤ 2 pp,
  inside the band, in both directions). It is kept at 64 for the default
  only because 128-token chunks need it.
- **bge-small does not lose enough to matter** at any chunk size; the
  440 MB default is chosen for the measured abstention floor and a 2–5 pp
  edge at 128 tokens, not because the small model is inadequate.
- **`token_f1` is a weak signal** (0.20–0.28 across every run): the
  models answer in full sentences and the references are three-to-eight
  word phrases, so F1 is bounded by verbosity. It orders the
  configurations the same way as `answer_rate` and `citation_precision`,
  which is all it is used for.
- The corpus is technical documentation with dense, distinct facts. Prose
  where the answer needs a whole section to be understood may favour
  larger chunks; that would be a second dataset with its own baseline,
  not a reason to distrust this one.
- Absolute numbers are for the exact NumPy store. An ANN backend
  (Phase G) gets its own baseline (plan.md §18 F7).
