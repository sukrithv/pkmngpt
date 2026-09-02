from unittest.mock import patch

from src.rag_llm import rag


def test_build_prompt_numbers_citations_from_one():
    contexts = [
        {"source": "pokeapi:pokemon:charizard", "text": "Fire/Flying type."},
        {"source": "pokeapi:type:fire", "text": "Strong against grass."},
    ]
    prompt = rag.build_prompt("What type is Charizard?", contexts)
    assert "[1] (source: pokeapi:pokemon:charizard)" in prompt
    assert "[2] (source: pokeapi:type:fire)" in prompt


def test_build_prompt_handles_empty_contexts():
    prompt = rag.build_prompt("What type is Charizard?", [])
    assert "Question: What type is Charizard?" in prompt  # doesn't crash on empty block


def test_answer_orchestrates_query_and_generate():
    fake_contexts = [{"source": "s", "text": "t", "score": 1.0}]
    with (
        patch.object(rag.pokeapi, "query", return_value=fake_contexts) as mock_query,
        patch.object(rag.llm, "generate", return_value="the answer") as mock_generate,
    ):
        result = rag.answer("What type is Charizard?", top_k=3)

    mock_query.assert_called_once_with("What type is Charizard?", top_k=3)
    assert mock_generate.call_args.kwargs["system"] == rag.SYSTEM_PROMPT
    assert result == {
        "question": "What type is Charizard?",
        "answer": "the answer",
        "contexts": fake_contexts,
    }


def test_answer_uses_default_top_k_from_config():
    with (
        patch.object(rag.pokeapi, "query", return_value=[]) as mock_query,
        patch.object(rag.llm, "generate", return_value=""),
    ):
        rag.answer("anything")
    assert mock_query.call_args.kwargs["top_k"] == rag.config.TOP_K
