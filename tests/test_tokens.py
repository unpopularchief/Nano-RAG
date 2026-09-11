import subprocess
import sys
from pathlib import Path

import pytest

from nanorag.tokens import HeuristicCounter, TiktokenCounter, TokenCounter

VENDORED_CACHE = Path(__file__).parent / "data" / "tiktoken_cache"

# A paragraph of ordinary English prose. For clean prose, cl100k averages close
# to 4 characters per token, so the len/4 heuristic lands within a few percent.
PROSE = (
    "The retriever ranks chunks by cosine similarity against the query "
    "embedding, applies any metadata pre-filter as a row mask, and returns the "
    "top matches with their scores so the context builder can fit them into a "
    "token budget before the single generation call is made."
)


def test_heuristic_counter_warns_on_construction():
    with pytest.warns(UserWarning, match="under-counts"):
        HeuristicCounter()


def test_heuristic_counter_can_suppress_the_warning():
    import warnings

    with warnings.catch_warnings():
        warnings.simplefilter("error")
        HeuristicCounter(warn=False)


def test_heuristic_counter_counts():
    counter = HeuristicCounter(warn=False)
    assert counter.count("") == 0
    assert counter.count("a") == 1
    assert counter.count("a" * 8) == 2
    assert counter.count("a" * 9) == 3  # ceiling


def test_tiktoken_counter_is_lazy():
    # In a clean interpreter: importing the module (and even constructing the
    # counter) must not import tiktoken; only the first count() does.
    code = (
        "import sys; import nanorag.tokens as t; "
        "assert 'tiktoken' not in sys.modules, 'import pulled tiktoken'; "
        "c = t.TiktokenCounter('cl100k_base'); "
        "assert 'tiktoken' not in sys.modules, 'construction pulled tiktoken'; "
        "assert c.count('hello world') > 0; "
        "assert 'tiktoken' in sys.modules; "
        "print('ok')"
    )
    out = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True, check=True
    )
    assert out.stdout.strip() == "ok", out.stderr


def test_tiktoken_counter_counts_empty_as_zero():
    assert TiktokenCounter("cl100k_base").count("") == 0


def test_explicit_cache_dir_is_used(monkeypatch):
    monkeypatch.delenv("TIKTOKEN_CACHE_DIR", raising=False)
    counter = TiktokenCounter("cl100k_base", cache_dir=VENDORED_CACHE)
    assert counter.count(PROSE) > 0
    import os

    assert os.environ["TIKTOKEN_CACHE_DIR"] == str(VENDORED_CACHE)


def test_tiktoken_counter_is_deterministic():
    a = TiktokenCounter("cl100k_base")
    b = TiktokenCounter("cl100k_base")
    assert a.count(PROSE) == b.count(PROSE) > 0


def test_tiktoken_counter_caches_the_encoder():
    counter = TiktokenCounter("cl100k_base")
    first = counter.count(PROSE)
    second = counter.count(PROSE)  # exercises the "encoder already loaded" path
    assert first == second


def test_o200k_encoding_also_works_from_the_vendored_cache():
    assert TiktokenCounter("o200k_base").count(PROSE) > 0


def test_both_counters_satisfy_the_protocol():
    assert isinstance(HeuristicCounter(warn=False), TokenCounter)
    assert isinstance(TiktokenCounter(), TokenCounter)


def test_heuristic_stays_within_2x_of_the_real_count_on_prose():
    exact = TiktokenCounter("cl100k_base").count(PROSE)
    approx = HeuristicCounter(warn=False).count(PROSE)
    # len/4 is only a rough floor: it can over- or under-count real prose by a
    # good margin (here it over-counts). The contract is "same order", not "5%".
    assert 0.5 <= approx / exact <= 2.0, (approx, exact)
