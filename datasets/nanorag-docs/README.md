# `nanorag-docs` — the Phase D evaluation set

A frozen corpus of technical prose and graded questions over it. This is
what `nanorag eval` and the `-m eval` regression gate run against; the
numbers in `docs/evaluation.md` and the thresholds in `thresholds.json`
are measured on exactly these bytes.

```
datasets/nanorag-docs/
├── corpus/
│   ├── nanorag/         7 of this project's own documents, as of session D2
│   └── third-party/    12 READMEs of permissively licensed Python packages
├── dev.jsonl            341 items: 317 answerable, 24 unanswerable
├── thresholds.json      committed baseline + tolerance per gated metric
└── README.md            this file
```

**The corpus is frozen.** The `nanorag/` documents are snapshots — they do
not track the live `README.md` etc., and must not be "updated"; a changed
corpus is a different dataset with a different baseline. Every file is
stored with LF line endings and NFC-normalised text, and the loader
normalises again on read, so a CRLF checkout on Windows produces the same
chunks (and the same numbers) as an LF checkout on the CI runner.

## Item format

One JSON object per line of `dev.jsonl` (schema and semantics in
`nanorag/evaluation/datasets.py`):

```json
{"id": "readme-011",
 "question": "In what order does from_defaults() pick a generator?",
 "gold": [{"source_uri": "nanorag/README.md",
           "quote": "a Groq key,\nthen a Gemini key, then a reachable Ollama"}],
 "answer": "Groq key, then Gemini key, then a reachable Ollama",
 "tags": ["readme", "generation"]}
```

`gold` locates the answer by **quote** — a verbatim excerpt of the named
document, matched with whitespace collapsed (so it may span a line wrap)
and required to occur exactly once. At load time each quote resolves to
character offsets in the document; a retrieved chunk is *relevant* when it
overlaps one of those spans. Gold is therefore defined on the document,
not on any chunker's output, which is what lets one dataset grade the
chunk-size sweep. A few items carry two gold spans (the answer needs both
places). `answer` is an optional short reference for the token-F1 answer
metric; `tags` name the source document and topic.

`"gold": []` marks an item **unanswerable** from this corpus. Twelve are
off-topic (baking, chess, gibberish) and twelve are *on-topic but
uncovered* — questions about these very projects whose answer is simply
not in the corpus (`nanorag`'s BM25 tokeniser, `httpx`'s pool size). The
second kind is the hard band the Gate C abstention measurement identified;
both grade `abstain_rate`, never ranking.

## How the items were written

Every item was written by hand (by the assistant, reviewed against the
source) while reading the corpus document, one question per distinct fact,
phrased as a user would ask rather than by copying the sentence. The
quote was then checked mechanically: `load_dataset()` refuses a quote that
is missing or ambiguous, and `tests/test_evaluation_dataset_nanorag_docs.py`
loads the whole set in the default suite. Two mistakes the process caught
and that a future contributor should expect: a quote that wraps a heading
or code fence boundary rarely survives, and a phrase repeated in a
changelog needs a longer quote to be unique.

Counts per source (answerable): README 34, CONTRIBUTING 10, ROADMAP 28,
CHANGELOG 44, conventions 15, providers 15, cli 15, charset-normalizer 20,
loguru 20, mmh3 12, fastembed 12, tokenizers 12, idna 13, tiktoken 10,
httpx 10, urllib3 6, httpcore 18, cfgv 15, platformdirs 8.

## Thresholds

`thresholds.json` holds, per gated metric, the **baseline** measured when
the file was committed and the **tolerance** below it that is not a
regression (`value < baseline − tolerance` fails). How each tolerance was
chosen, the sweep that produced the baseline configuration, and the
numbers themselves are in `docs/evaluation.md`. To re-baseline after a
deliberate change, run `nanorag eval datasets/nanorag-docs/dev.jsonl` on
the CI runner image and copy the metrics in — never lower a floor to make
a red run green.

## Third-party documents — provenance and licences

The `corpus/third-party/` files are the long descriptions (READMEs) of
the following packages, copied verbatim from the installed package
metadata at the versions listed and converted to LF/NFC. They are
included solely as evaluation text; each remains under its own licence,
whose notice is reproduced here as those licences require. Nothing else
from these packages is redistributed.

| File | Package / version | Licence | Copyright | Source |
| --- | --- | --- | --- | --- |
| `charset-normalizer.md` | charset-normalizer 3.5.1 | MIT | © 2025 TAHRI Ahmed R. | <https://github.com/jawah/charset_normalizer> |
| `httpcore.md` | httpcore 1.0.9 | BSD-3-Clause | © 2020 Encode OSS Ltd | <https://github.com/encode/httpcore> |
| `httpx.md` | httpx 0.28.1 | BSD-3-Clause | © 2019 Encode OSS Ltd | <https://github.com/encode/httpx> |
| `loguru.md` | loguru 0.7.3 | MIT | © 2017 Delgan | <https://github.com/Delgan/loguru> |
| `mmh3.md` | mmh3 5.3.0 | MIT | © 2011–2026 Hajime Senuma | <https://github.com/hajimes/mmh3> |
| `fastembed.md` | fastembed 0.8.0 | Apache-2.0 | © Qdrant | <https://github.com/qdrant/fastembed> |
| `tokenizers.md` | tokenizers 0.23.2 | Apache-2.0 | © Hugging Face | <https://github.com/huggingface/tokenizers> |
| `idna.md` | idna 3.19 | BSD-3-Clause | © 2013–2026 Kim Davies and contributors | <https://github.com/kjd/idna> |
| `tiktoken.md` | tiktoken 0.14.0 | MIT | © 2022 OpenAI, Shantanu Jain | <https://github.com/openai/tiktoken> |
| `urllib3.md` | urllib3 2.7.0 | MIT | © 2008–2020 Andrey Petrov and contributors | <https://github.com/urllib3/urllib3> |
| `cfgv.md` | cfgv 3.5.0 | MIT | © 2018 Anthony Sottile | <https://github.com/asottile/cfgv> |
| `platformdirs.md` | platformdirs 4.11.8 | MIT | © 2010–202x The platformdirs developers | <https://github.com/tox-dev/platformdirs> |

The MIT and BSD-3-Clause texts, and the Apache License 2.0, are reproduced
in each package's own distribution (`*.dist-info/licenses/`) and at the
source links above. The `nanorag/` documents are this project's own, MIT.
