# Vendored tiktoken encoding cache

Committed on purpose. `tests/conftest.py` points `TIKTOKEN_CACHE_DIR` here so
the default test suite never downloads a tokenizer at runtime (plan.md §16 A1,
Gate A checklist: `pytest -q` must pass with sockets blocked).

The blobs are tiktoken's own cache files, keyed by a hash of the vocab URL:

| File | Encoding |
| --- | --- |
| `9b5ad71b2ce5302211f9c61530b329a4922fc6a4` | `cl100k_base` |
| `fb374d419588a4632f3f557e76b4b70aebbca790` | `o200k_base` |

To refresh: delete a blob, run any test that builds a `TiktokenCounter` with
network access once, commit the regenerated file. A change here must be
deliberate and reviewed.
