import unicodedata

from nanorag.cleaning import clean_text

E_ACUTE_COMPOSED = chr(0x00E9)
E_ACUTE_DECOMPOSED = unicodedata.normalize("NFD", E_ACUTE_COMPOSED)


def test_clean_text_collapses_line_endings():
    assert clean_text("a\r\nb\rc\nd") == "a\nb\nc\nd"


def test_clean_text_applies_nfc():
    assert clean_text(E_ACUTE_DECOMPOSED) == E_ACUTE_COMPOSED


def test_clean_text_strips_trailing_whitespace_per_line():
    assert clean_text("a   \nb\t\n") == "a\nb"


def test_clean_text_collapses_blank_line_runs():
    assert clean_text("a\n\n\n\n\nb") == "a\n\nb"


def test_clean_text_keeps_single_blank_line():
    assert clean_text("a\n\nb") == "a\n\nb"


def test_clean_text_trims_leading_and_trailing_blank_lines():
    assert clean_text("\n\n\nhello\n\n\n") == "hello"


def test_clean_text_empty_string():
    assert clean_text("") == ""


def test_clean_text_is_idempotent():
    text = "  a  \r\n\r\n\r\n\r\nb  \r\n" + E_ACUTE_DECOMPOSED
    once = clean_text(text)
    assert clean_text(once) == once


def test_clean_text_does_not_touch_internal_single_spaces():
    assert clean_text("a  b   c") == "a  b   c"
