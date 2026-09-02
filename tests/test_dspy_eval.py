import json
from unittest.mock import MagicMock, patch

import pytest

from src import dspy_eval


def test_configure_dspy_builds_ollama_lm(monkeypatch):
    monkeypatch.setattr(dspy_eval.config, "LLM_BACKEND", "ollama")
    with (
        patch.object(dspy_eval.dspy, "LM") as mock_lm,
        patch.object(dspy_eval.dspy, "configure") as mock_configure,
    ):
        dspy_eval.configure_dspy()
    assert mock_lm.call_args.args[0].startswith("ollama_chat/")
    mock_configure.assert_called_once()


def test_configure_dspy_builds_claude_lm(monkeypatch):
    monkeypatch.setattr(dspy_eval.config, "LLM_BACKEND", "claude")
    with (
        patch.object(dspy_eval.dspy, "LM") as mock_lm,
        patch.object(dspy_eval.dspy, "configure"),
    ):
        dspy_eval.configure_dspy()
    assert mock_lm.call_args.args[0].startswith("anthropic/")


def test_configure_dspy_raises_on_unknown_backend(monkeypatch):
    monkeypatch.setattr(dspy_eval.config, "LLM_BACKEND", "bogus")
    with pytest.raises(ValueError):
        dspy_eval.configure_dspy()


def test_keyword_metric_all_keywords_present():
    example = MagicMock(keywords=["fire", "flying"])
    prediction = MagicMock(answer="Charizard is a Fire/Flying type.")
    assert dspy_eval.keyword_metric(example, prediction) is True


def test_keyword_metric_missing_keyword():
    example = MagicMock(keywords=["fire", "dragon"])
    prediction = MagicMock(answer="Charizard is a Fire/Flying type.")
    assert dspy_eval.keyword_metric(example, prediction) is False


def test_keyword_metric_is_case_insensitive():
    example = MagicMock(keywords=["FIRE"])
    prediction = MagicMock(answer="charizard is a fire type")
    assert dspy_eval.keyword_metric(example, prediction) is True


def test_keyword_metric_defaults_to_empty_list_when_absent():
    example = MagicMock(spec=[])
    prediction = MagicMock(answer="anything")
    assert dspy_eval.keyword_metric(example, prediction) is True


def test_load_devset_raises_when_file_missing(monkeypatch, tmp_path):
    monkeypatch.setattr(dspy_eval, "EVAL_SET_PATH", tmp_path / "missing.json")
    with pytest.raises(FileNotFoundError):
        dspy_eval.load_devset()


def test_load_devset_defaults_missing_fields(monkeypatch, tmp_path):
    path = tmp_path / "eval_set.json"
    path.write_text(json.dumps([{"question": "What type is Pikachu?"}]))
    monkeypatch.setattr(dspy_eval, "EVAL_SET_PATH", path)
    examples = dspy_eval.load_devset()
    assert examples[0].answer == ""
    assert examples[0].keywords == []


def test_load_devset_marks_only_question_as_input(monkeypatch, tmp_path):
    path = tmp_path / "eval_set.json"
    path.write_text(json.dumps([{"question": "q", "answer": "a", "keywords": ["k"]}]))
    monkeypatch.setattr(dspy_eval, "EVAL_SET_PATH", path)
    example = dspy_eval.load_devset()[0]
    assert set(example.inputs().keys()) == {"question"}


def test_run_eval_silently_falls_back_to_judge_on_unknown_metric_name(monkeypatch):
    monkeypatch.setattr(dspy_eval, "configure_dspy", lambda: None)
    monkeypatch.setattr(dspy_eval, "load_devset", list)
    with (
        patch.object(dspy_eval, "make_llm_judge_metric") as mock_judge,
        patch.object(dspy_eval.dspy.evaluate, "Evaluate") as mock_evaluate,
    ):
        mock_evaluate.return_value = lambda program: 0.0
        dspy_eval.run_eval("keywrod")
    mock_judge.assert_called_once()


def test_pokemon_rag_forward_builds_context_and_attaches_hits():
    hits = [
        {"source": "pokeapi:pokemon:charizard", "text": "Fire/Flying.", "score": 1.0}
    ]
    with (
        patch.object(dspy_eval.pokeapi, "query", return_value=hits),
        patch.object(dspy_eval.dspy, "Predict") as mock_predict_cls,
    ):
        mock_predict_cls.return_value.return_value = MagicMock()
        program = dspy_eval.PokemonRAG()
        result = program.forward("What type is Charizard?")

    call_kwargs = program.generate_answer.call_args.kwargs
    assert "(pokeapi:pokemon:charizard) Fire/Flying." in call_kwargs["context"]
    assert result.contexts == hits


def test_pokemon_rag_forward_handles_empty_hits():
    with (
        patch.object(dspy_eval.pokeapi, "query", return_value=[]),
        patch.object(dspy_eval.dspy, "Predict") as mock_predict_cls,
    ):
        mock_predict_cls.return_value.return_value = MagicMock()
        program = dspy_eval.PokemonRAG()
        program.forward("anything")
    assert program.generate_answer.call_args.kwargs["context"] == ""
