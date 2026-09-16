"""Per-answer metrics — pure functions over a public ``Answer``.

What can be graded without a judge model, keyless and offline:

- **citation validity** — the share of ``[n]`` markers in the text that
  resolved to a context block (the D1 validity rate, recomputed from the
  public ``Answer`` alone).
- **citation precision** — the share of resolved citations whose span
  overlaps a gold span: did the model cite *where the answer is*, not
  merely something that was in the prompt.
- **abstained** — whether the answer is the fixed abstention text. Graded
  against the item: an answerable question should be answered, an
  unanswerable one abstained on.
- **token F1** — SQuAD-style bag-of-tokens F1 against a short reference
  answer (lower-cased, punctuation and articles stripped). A coarse lexical
  signal, not a judgement of correctness; it is only computed for items
  that carry a reference ``answer``.

A citation proves provenance, not truth (plan.md §11), and none of these
metrics claims otherwise.
"""

from __future__ import annotations

import re
import string
from collections import Counter
from collections.abc import Sequence

from nanorag.citations.parser import MARKER_PATTERN
from nanorag.evaluation.retrieval_metrics import coverage_of
from nanorag.prompting.templates import INSUFFICIENT_CONTEXT_TEXT
from nanorag.types import Answer

_ARTICLES = re.compile(r"\b(a|an|the)\b")
_PUNCTUATION = str.maketrans("", "", string.punctuation)


def citation_validity(answer: Answer) -> float | None:
    """Share of markers in ``answer.text`` that resolved; ``None`` if no markers."""
    labels = {c.label for c in answer.citations}
    total = resolved = 0
    for match in MARKER_PATTERN.finditer(answer.text):
        total += 1
        if int(match.group(1) or match.group(2)) in labels:
            resolved += 1
    return resolved / total if total else None


def citation_precision(
    answer: Answer, gold_spans: Sequence[tuple[str, int, int]]
) -> float | None:
    """Share of resolved citations overlapping a gold span; ``None`` if none cited."""
    if not answer.citations:
        return None
    on_gold = sum(
        1
        for c in answer.citations
        if coverage_of(c.doc_id, c.start_char, c.end_char, gold_spans)
    )
    return on_gold / len(answer.citations)


def abstained(answer: Answer) -> bool:
    """Return True when the answer is the fixed abstention text (modulo whitespace)."""
    return " ".join(answer.text.split()) == " ".join(INSUFFICIENT_CONTEXT_TEXT.split())


def normalize_answer(text: str) -> list[str]:
    """SQuAD normalisation: lower-case, drop punctuation and articles, split."""
    lowered = text.lower().translate(_PUNCTUATION)
    return _ARTICLES.sub(" ", lowered).split()


def token_f1(prediction: str, reference: str) -> float:
    """Bag-of-tokens F1 between *prediction* and *reference* after normalisation."""
    predicted = normalize_answer(prediction)
    expected = normalize_answer(reference)
    if not predicted or not expected:
        return 1.0 if predicted == expected else 0.0
    overlap = sum((Counter(predicted) & Counter(expected)).values())
    if overlap == 0:
        return 0.0
    precision = overlap / len(predicted)
    recall = overlap / len(expected)
    return 2 * precision * recall / (precision + recall)
