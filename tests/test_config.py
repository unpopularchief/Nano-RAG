import dataclasses
import textwrap

import pytest

from nanorag.config import Settings
from nanorag.errors import ConfigError

ENV_VARS = [
    "NANORAG_PROFILE",
    "NANORAG_PERSIST_DIR",
    "NANORAG_EMBEDDING_MODEL",
    "NANORAG_GENERATOR_PRESET",
    "NANORAG_LOG_LEVEL",
    "NANORAG_REQUEST_TIMEOUT_S",
    "NANORAG_CONNECT_TIMEOUT_S",
    "NANORAG_MAX_RETRIES",
    "NANORAG_RETRY_BASE_DELAY_S",
    "NANORAG_EMBED_BATCH_SIZE",
]


@pytest.fixture(autouse=True)
def _clean_env(monkeypatch):
    for name in ENV_VARS:
        monkeypatch.delenv(name, raising=False)


@pytest.fixture
def empty_root(tmp_path):
    # A directory with no pyproject.toml, so `load` sees only defaults.
    return tmp_path


def write_pyproject(root, body: str) -> None:
    (root / "pyproject.toml").write_text(textwrap.dedent(body), encoding="utf-8")


# --- defaults & construction -------------------------------------------


def test_defaults(empty_root):
    s = Settings.load(project_root=empty_root)
    assert s.persist_dir == ".nanorag"
    assert s.generator_preset == "auto"
    assert s.max_retries == 3
    assert s.embed_batch_size == 64


def test_settings_is_frozen():
    s = Settings()
    with pytest.raises(dataclasses.FrozenInstanceError):
        s.log_level = "DEBUG"


def test_to_dict_covers_every_field():
    d = Settings().to_dict()
    assert d["persist_dir"] == ".nanorag"
    assert set(d) == {f.name for f in dataclasses.fields(Settings)}


# --- validation --------------------------------------------------------


@pytest.mark.parametrize(
    "kwargs",
    [
        {"log_level": "LOUD"},
        {"generator_preset": "openai"},
        {"request_timeout_s": 0},
        {"connect_timeout_s": -1.0},
        {"retry_base_delay_s": 0.0},
        {"max_retries": -1},
        {"embed_batch_size": 0},
        {"persist_dir": ""},
        {"embedding_model": ""},
    ],
)
def test_invalid_values_raise_config_error(kwargs):
    with pytest.raises(ConfigError):
        Settings(**kwargs)


# --- precedence -------------------------------------------------------


def test_pyproject_overrides_defaults(empty_root):
    write_pyproject(
        empty_root,
        """
        [tool.nanorag]
        persist_dir = "from_toml"
        max_retries = 7
        """,
    )
    s = Settings.load(project_root=empty_root)
    assert s.persist_dir == "from_toml"
    assert s.max_retries == 7


def test_env_overrides_pyproject(empty_root, monkeypatch):
    write_pyproject(empty_root, '[tool.nanorag]\npersist_dir = "from_toml"\n')
    monkeypatch.setenv("NANORAG_PERSIST_DIR", "from_env")
    monkeypatch.setenv("NANORAG_MAX_RETRIES", "9")
    s = Settings.load(project_root=empty_root)
    assert s.persist_dir == "from_env"
    assert s.max_retries == 9


def test_explicit_override_wins_over_env(empty_root, monkeypatch):
    monkeypatch.setenv("NANORAG_PERSIST_DIR", "from_env")
    s = Settings.load(project_root=empty_root, persist_dir="explicit")
    assert s.persist_dir == "explicit"


def test_profile_applies_and_env_overrides_a_profile_value(empty_root, monkeypatch):
    s = Settings.load(project_root=empty_root, profile="local")
    assert s.generator_preset == "ollama"

    monkeypatch.setenv("NANORAG_GENERATOR_PRESET", "groq")
    s2 = Settings.load(project_root=empty_root, profile="local")
    assert s2.generator_preset == "groq"


def test_profile_from_env_var(empty_root, monkeypatch):
    monkeypatch.setenv("NANORAG_PROFILE", "local")
    assert Settings.load(project_root=empty_root).generator_preset == "ollama"


# --- error paths -----------------------------------------------------


def test_unknown_profile_raises(empty_root):
    with pytest.raises(ConfigError, match="unknown profile"):
        Settings.load(project_root=empty_root, profile="prod")


def test_unknown_override_raises(empty_root):
    with pytest.raises(ConfigError, match="unknown setting"):
        Settings.load(project_root=empty_root, retriez=1)


def test_unknown_pyproject_key_raises(empty_root):
    write_pyproject(empty_root, "[tool.nanorag]\nnope = 1\n")
    with pytest.raises(ConfigError, match="unknown \\[tool.nanorag\\] keys"):
        Settings.load(project_root=empty_root)


def test_malformed_env_value_raises(empty_root, monkeypatch):
    monkeypatch.setenv("NANORAG_MAX_RETRIES", "lots")
    with pytest.raises(ConfigError, match="invalid value for NANORAG_MAX_RETRIES"):
        Settings.load(project_root=empty_root)


def test_non_table_tool_nanorag_raises(empty_root):
    write_pyproject(empty_root, '[tool]\nnanorag = "oops"\n')
    with pytest.raises(ConfigError, match="must be a table"):
        Settings.load(project_root=empty_root)


def test_float_and_int_env_coercion(empty_root, monkeypatch):
    monkeypatch.setenv("NANORAG_REQUEST_TIMEOUT_S", "12.5")
    monkeypatch.setenv("NANORAG_EMBED_BATCH_SIZE", "128")
    s = Settings.load(project_root=empty_root)
    assert s.request_timeout_s == 12.5
    assert s.embed_batch_size == 128
