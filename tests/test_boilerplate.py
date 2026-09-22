import pytest

from nanorag.cleaning import strip_boilerplate


def test_repeated_page_edge_lines_are_removed_conservatively():
    text = (
        "Nano RAG manual\nFirst body\nPage footer\f"
        "Nano RAG manual\nSecond body\nPage footer\f"
        "Nano RAG manual\nThird body\nPage footer"
    )

    cleaned = strip_boilerplate(text, edge_lines=1, min_repetitions=3)

    assert "Nano RAG manual" not in cleaned
    assert "Page footer" not in cleaned
    assert "First body\fSecond body\fThird body" == cleaned


def test_repetition_in_body_is_not_removed_when_it_is_not_at_an_edge():
    text = "Top A\nShared body\nBottom A\fTop B\nShared body\nBottom B"
    assert strip_boilerplate(text, min_repetitions=2, edge_lines=1) == text


def test_declared_html_navigation_and_footer_phrases_are_removed():
    text = "Documentation\nArticle body\nPrivacy policy"

    cleaned = strip_boilerplate(text, phrases={"Documentation", "Privacy policy"})

    assert cleaned == "Article body"


def test_retained_lines_are_not_normalized_or_rewritten():
    text = "  exact spacing  \nRemove me\nkept\tline"
    cleaned = strip_boilerplate(text, phrases=["Remove me"])
    assert cleaned == "  exact spacing  \nkept\tline"


def test_boilerplate_stripping_is_idempotent():
    text = "Header\nOne\fHeader\nTwo\fHeader\nThree"
    once = strip_boilerplate(text, edge_lines=1)
    assert strip_boilerplate(once, edge_lines=1) == once


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [({"edge_lines": -1}, "edge_lines"), ({"min_repetitions": 1}, "min_repetitions")],
)
def test_boilerplate_options_are_validated(kwargs, message):
    with pytest.raises(ValueError, match=message):
        strip_boilerplate("text", **kwargs)
