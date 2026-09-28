# Deployment (Phase H1)

Nano RAG runs as an embedded library or in one long-running process. Keep one
writer for a corpus. File-backed SQLite connections use WAL, so separate
connections can read while a writer commits. A `SqliteDocumentStore` connection
belongs to the thread that created it; give each thread its own connection.
Share one `NumpyVectorStore` between those threads and coordinate document
writes with retrieval at the application boundary. A query must not search the
matrix and then hydrate chunks while another thread replaces or deletes those
chunks in SQLite. The matrix has a writer lock and snapshot reads, but that
does not make a two-store operation atomic. SQLite commits before matrix
mutation; after a crash, opening from SQLite repairs the index. Back up SQLite
with its backup API or a stopped writer, not by copying only the main database
file while WAL writes are active.

Do not run multiple workers or instances against one index. Each would have
its own matrix and provider rate limiter. Qdrant or pgvector changes vector
storage, but does not make the limiter or other state cross-process. Serverless
cold starts rebuild the index and may download the embedding model, so they
are outside the supported deployment scope through 1.0.

## Offline image and cold start

Pin dependencies with `uv.lock` and choose a fixed local embedding model. In
the image build, instantiate `FastEmbedEmbedder(model_id=..., cache_dir=...)`
to populate a cache directory, record SHA-256 checksums of the downloaded
model files, and verify those checksums during the build. Copy the verified
cache into the runtime image at the same path. Start the application with that
`cache_dir` and block outbound access in a deployment test to prove the model
is fully present. `Rag.from_defaults()` uses FastEmbed's default cache
resolution, so an explicit embedder is preferable when image paths are fixed.
Provider API keys belong in runtime environment variables, never the image.

An optional NumPy snapshot skips reading every SQLite embedding BLOB on a
normal cold start:

```python
from pathlib import Path

from nanorag import Rag

root = Path("/data/nanorag")
rag = Rag.from_defaults(persist_dir=root, snapshot_path=root / "matrix.npy")
# After a completed ingest or sync, while writes are quiescent:
rag.save_snapshot()
```

`matrix.npy` has a JSON manifest with chunk ids, model/dimension, a durable
embedding revision, and a SHA-256 digest. Startup accepts it only when these
match SQLite; a missing, damaged or stale snapshot falls back to a full
rebuild. Any embedding insert, overwrite or delete changes the revision, so
save again after writes if faster subsequent starts matter. SQLite remains the
source of truth. The digest check reads the `.npy` file once; the speedup is
avoiding SQLite row decoding and matrix assembly, not avoiding disk I/O.

## Logs, events and cost

The CLI emits JSON log lines on stderr through `JsonFormatter`. Applications
can attach `nanorag.observability.JsonFormatter` to their own handlers. It
masks credential assignments, Bearer tokens and values of configured secret
environment variables. Lifecycle events contain only counts and cost, never
source text, questions, prompts or answers. Pass `event_hook=callable` to
`Rag` for `ingest_complete` and `query_complete` events; a hook failure
produces a warning and does not fail the operation. Avoid logging returned
`Answer` objects or exception traces in your own handlers if they can contain
document or provider response text.

Provider-reported prompt, completion and total token counts are available on
`Answer.usage`. If a provider omits them, the configured token counter supplies
an estimate. Pass `token_prices={(provider, model):
(input_usd_per_million, output_usd_per_million)}` to `Rag` for cost estimates.
Rates are explicit because provider prices change. `cost_usd=0` without a
configured rate means **unpriced**, not necessarily free. The exact served
provider and model are used after fallback; reported token counts remain the
source for the estimate when available.
