import importlib

import pytest


def _reload_config(monkeypatch, **env):
    for key, value in env.items():
        monkeypatch.setenv(key, value)
    import config

    return importlib.reload(config)


def test_defaults_when_env_unset(monkeypatch):
    monkeypatch.delenv("LLM_BACKEND", raising=False)
    monkeypatch.delenv("TOP_K", raising=False)
    cfg = _reload_config(monkeypatch)
    assert cfg.LLM_BACKEND == "ollama"
    assert cfg.TOP_K == 5


def test_env_override(monkeypatch):
    cfg = _reload_config(monkeypatch, LLM_BACKEND="claude", TOP_K="10")
    assert cfg.LLM_BACKEND == "claude"
    assert cfg.TOP_K == 10


def test_llm_backend_lowercased(monkeypatch):
    cfg = _reload_config(monkeypatch, LLM_BACKEND="OLLAMA")
    assert cfg.LLM_BACKEND == "ollama"


def test_invalid_numeric_env_raises(monkeypatch):
    monkeypatch.setenv("TOP_K", "not-a-number")
    with pytest.raises(ValueError):
        _reload_config(monkeypatch)


def test_paths_resolve_relative_to_root_dir(monkeypatch):
    cfg = _reload_config(monkeypatch)
    assert str(cfg.DATA_RAW_DIR).startswith(str(cfg.ROOT_DIR))
