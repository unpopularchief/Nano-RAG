"""``evaluation/datasets.py``: item parsing and validation, corpus loading with
line-ending normalisation, quote resolution (whitespace-insensitive, must be
unique), and the loud failures a malformed dataset gets."""

import json

import pytest

from nanorag.errors import EvaluationError
from nanorag.evaluation import (
    EvalItem,
    GoldSpan,
    load_corpus,
    load_dataset,
    load_items,
    resolve_spans,
)

CORPUS = {
    "cats.md": "# Cats\n\nCats sleep most of\nthe day. A cat purrs when content.\n",
    "dogs.txt": "Dogs bark at strangers.\r\nA dog wags its tail when happy.\r\n",
    "sub/fish.md": "Fish swim in water. Fish swim in water.\n",
}


def _write(root, items):
    corpus = root / "corpus"
    for name, body in CORPUS.items():
        path = corpus / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(body.encode("utf-8"))
    (root / "dev.jsonl").write_text(
        "\n".join(json.dumps(i) for i in items) + "\n", encoding="utf-8"
    )
    return root / "dev.jsonl"


def _item(id_, question="q?", gold=None, **extra):
    data = {"id": id_, "question": question, "gold": gold or []}
    data.update(extra)
    return data


def test_load_dataset_resolves_quotes_across_line_wraps_and_crlf(tmp_path):
    path = _write(
        tmp_path,
        [
            _item(
                "a",
                "When do cats sleep?",
                [{"source_uri": "cats.md", "quote": "sleep most of the day"}],
                answer="most of the day",
                tags=["cats"],
            ),
            _item("b", "Dogs?", [{"source_uri": "dogs.txt", "quote": "wags its tail"}]),
            _item("c", "Unanswerable?"),
        ],
    )
    dataset = load_dataset(path)
    assert dataset.name == tmp_path.name
    assert [i.id for i in dataset.items] == ["a", "b", "c"]
    assert [i.id for i in dataset.answerable] == ["a", "b"]
    assert [i.id for i in dataset.unanswerable] == ["c"]
    assert dataset.items[0].answer == "most of the day"
    assert dataset.items[0].tags == ("cats",)
    # sorted by source_uri, text normalised to LF regardless of the bytes
    assert [d.source_uri for d in dataset.documents] == [
        "cats.md",
        "dogs.txt",
        "sub/fish.md",
    ]
    dogs = dataset.documents[1]
    assert "\r" not in dogs.text
    (span,) = dataset.spans["a"]
    cats = dataset.documents[0]
    assert cats.text[span.start_char : span.end_char] == "sleep most of\nthe day"
    (span,) = dataset.spans["b"]
    assert dogs.text[span.start_char : span.end_char] == "wags its tail"
    assert span.doc_id == dogs.doc_id and span.source_uri == "dogs.txt"
    assert span.as_triple() == (dogs.doc_id, span.start_char, span.end_char)
    assert dataset.spans["c"] == ()


def test_explicit_corpus_dir_and_missing_quote(tmp_path):
    path = _write(
        tmp_path, [_item("a", gold=[{"source_uri": "cats.md", "quote": "nope"}])]
    )
    with pytest.raises(EvaluationError, match="item 'a': quote not found"):
        load_dataset(path, tmp_path / "corpus")


def test_ambiguous_quote_and_unknown_source_fail_loudly(tmp_path):
    documents = load_corpus(_write(tmp_path, [_item("x")]).parent / "corpus")
    dup = EvalItem("dup", "q", (GoldSpan("sub/fish.md", "Fish swim"),))
    with pytest.raises(EvaluationError, match="found 2 times"):
        resolve_spans([dup], documents)
    missing = EvalItem("m", "q", (GoldSpan("birds.md", "tweet"),))
    with pytest.raises(EvaluationError, match="not in the corpus"):
        resolve_spans([missing], documents)


@pytest.mark.parametrize(
    "lines, message",
    [
        (["{not json"], "not valid JSON"),
        (["[1, 2]"], "expected a JSON object"),
        ([json.dumps({"id": "a"})], "'question'"),
        ([json.dumps({"id": "a", "question": "q", "extra": 1})], "unknown keys"),
        ([json.dumps({"id": "a", "question": "q", "gold": "x"})], "gold must be"),
        ([json.dumps({"id": "a", "question": "q", "answer": 3})], "answer must be"),
        ([json.dumps({"id": "a", "question": "q", "tags": "t"})], "tags must be"),
        ([json.dumps({"id": "", "question": "q"})], "non-empty"),
        ([json.dumps({"id": "a", "question": "  "})], "non-blank"),
        (
            [json.dumps({"id": "a", "question": "q", "gold": [{"source_uri": "s"}]})],
            "'quote'",
        ),
        ([json.dumps(_item("a")), json.dumps(_item("a"))], "duplicate item id"),
        ([""], "holds no items"),
    ],
)
def test_malformed_items_are_rejected_with_the_line_number(tmp_path, lines, message):
    path = tmp_path / "dev.jsonl"
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    with pytest.raises(EvaluationError, match=message):
        load_items(path)


def test_blank_lines_are_skipped_and_missing_files_named(tmp_path):
    path = tmp_path / "dev.jsonl"
    path.write_text("\n" + json.dumps(_item("a")) + "\n\n", encoding="utf-8")
    assert [i.id for i in load_items(path)] == ["a"]
    with pytest.raises(EvaluationError, match="no such dataset file"):
        load_items(tmp_path / "absent.jsonl")
    with pytest.raises(EvaluationError, match="no such corpus directory"):
        load_corpus(tmp_path / "absent")
    (tmp_path / "empty").mkdir()
    with pytest.raises(EvaluationError, match="holds no documents"):
        load_corpus(tmp_path / "empty")


def test_a_corpus_file_that_fails_to_load_is_an_error(tmp_path):
    corpus = tmp_path / "corpus"
    corpus.mkdir()
    (corpus / "ok.txt").write_text("fine", encoding="utf-8")
    (corpus / "bad.txt").write_bytes(b"\xff\xfe\x00\x00 not decodable \x80\x81")
    with pytest.raises(EvaluationError, match="failed to load: bad.txt"):
        load_corpus(corpus)


def test_value_objects_validate():
    with pytest.raises(ValueError):
        GoldSpan("", "q")
    with pytest.raises(ValueError):
        GoldSpan("a.md", "   ")
    with pytest.raises(ValueError):
        EvalItem("", "q")
    assert not EvalItem("a", "q").answerable
    from nanorag.evaluation import ResolvedSpan

    with pytest.raises(ValueError):
        ResolvedSpan("d", "s", 5, 5)
