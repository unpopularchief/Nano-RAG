from nanorag.loaders.globbing import glob_match


def test_star_matches_bare_filename():
    assert glob_match("a.txt", "*")


def test_star_does_not_cross_segment():
    assert not glob_match("a/b", "*")


def test_double_star_slash_star_matches_top_level_and_nested():
    assert glob_match("a.md", "**/*.md")
    assert glob_match("x/a.md", "**/*.md")
    assert glob_match("x/y/a.md", "**/*.md")


def test_single_star_glob_is_top_level_only():
    assert glob_match("a.md", "*.md")
    assert not glob_match("x/a.md", "*.md")


def test_bare_double_star_matches_everything_including_nested():
    assert glob_match("a.txt", "**")
    assert glob_match("x/y/a.txt", "**")


def test_dir_double_star_matches_direct_and_nested_children():
    assert glob_match("dir/a.txt", "dir/**")
    assert glob_match("dir/sub/a.txt", "dir/**")


def test_question_mark_matches_exactly_one_non_slash_character():
    assert glob_match("a", "?")
    assert not glob_match("ab", "?")
    assert not glob_match("a/", "?")


def test_literal_regex_special_characters_are_matched_literally():
    assert glob_match("a[1].txt", "a[1].txt")
    assert not glob_match("a1.txt", "a[1].txt")
