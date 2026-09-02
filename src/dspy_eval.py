"""DSPy evaluation harness for the Pokemon RAG pipeline.

Two things live here:
1. `PokemonRAG` — a dspy.Module version of the RAG pipeline (retrieve, then
   dspy.Predict for generation). Writing it as a real dspy.Module, rather
   than just calling src.llm.generate directly, is what lets you later run
   a DSPy optimizer (e.g. BootstrapFewShot) against it to auto-tune the
   prompt — not just measure it.
2. An evaluation metric + runner that scores the module against
   eval/eval_set.json using dspy.evaluate.Evaluate.

Usage:
    python -m src.dspy_eval
"""

import json
import re
from pathlib import Path

import dspy

import config
from src import pokeapi

EVAL_SET_PATH = config.ROOT_DIR / "eval" / "eval_set.json"


def configure_dspy():
    """Point DSPy's LM at whichever backend config.py selected."""
    if config.LLM_BACKEND == "ollama":
        lm = dspy.LM(
            f"ollama_chat/{config.OLLAMA_MODEL}",
            api_base=config.OLLAMA_BASE_URL,
            api_key="",
            cache=False,
        )
    elif config.LLM_BACKEND == "claude":
        lm = dspy.LM(
            f"anthropic/{config.ANTHROPIC_MODEL}",
            api_key=config.ANTHROPIC_API_KEY,
            cache=False,
        )
    else:
        raise ValueError(f"Unknown LLM_BACKEND: {config.LLM_BACKEND!r}")

    dspy.configure(lm=lm)
    return lm


class PokemonQA(dspy.Signature):
    """Answer a Pokemon question using only the given context."""

    context = dspy.InputField(desc="retrieved passages about Pokemon")
    question = dspy.InputField()
    answer = dspy.OutputField(desc="concise, factual answer grounded in the context")


class PokemonRAG(dspy.Module):
    def __init__(self, top_k=config.TOP_K):
        super().__init__()
        self.top_k = top_k
        self.generate_answer = dspy.Predict(PokemonQA)

    def forward(self, question):
        hits = pokeapi.query(question, top_k=self.top_k)
        context = "\n\n".join(f"({h['source']}) {h['text']}" for h in hits)
        prediction = self.generate_answer(context=context, question=question)
        prediction.contexts = hits
        return prediction


class JudgeCorrectness(dspy.Signature):
    """Judge whether a predicted answer is factually consistent with the gold answer."""

    question = dspy.InputField()
    gold_answer = dspy.InputField()
    predicted_answer = dspy.InputField()
    is_correct: bool = dspy.OutputField(
        desc="True if predicted_answer conveys the same facts as gold_answer"
    )


def make_llm_judge_metric():
    """LLM-as-judge metric: asks the configured LM whether the answer is correct.

    Use this once you're past simple keyword checks and want to score
    free-form answers.
    """
    judge = dspy.Predict(JudgeCorrectness)

    def metric(example, prediction, trace=None):
        result = judge(
            question=example.question,
            gold_answer=example.answer,
            predicted_answer=prediction.answer,
        )
        return bool(result.is_correct)

    return metric


def _normalize(text: str) -> str:
    """Lowercase and strip commas/whitespace."""
    return re.sub(r"[,\s]", "", text.lower())


def keyword_metric(
    example: dspy.Example, prediction: dspy.Prediction, trace=None
) -> bool:
    """Cheap, deterministic metric: did the answer mention every expected keyword?

    Normalizes both sides (lowercase, strips commas and whitespace) before the
    substring check so comma-formatted numbers and spacing differences don't cause false failures.
    """
    predicted = _normalize(prediction.answer)
    keywords = getattr(example, "keywords", [])
    return all(_normalize(kw) in predicted for kw in keywords)


def load_devset(paths: list[Path] = [EVAL_SET_PATH]) -> list[list[dspy.Example]]:

    examples = []
    for path in paths:
        if not path.exists():
            raise FileNotFoundError(f"{path} not found")
        raw = json.loads(path.read_text(encoding="utf-8"))
        subexamples = []
        for item in raw:
            example = dspy.Example(
                question=item["question"],
                answer=item.get("answer", ""),
                keywords=item.get("keywords", []),
            ).with_inputs("question")
            subexamples.append(example)
        examples.append(subexamples)

    return examples


def run_eval(metric_name="keyword"):
    configure_dspy()
    devset = load_devset()
    program = PokemonRAG()

    metric = keyword_metric if metric_name == "keyword" else make_llm_judge_metric()

    evaluator = dspy.evaluate.Evaluate(
        devset=devset,
        metric=metric,
        num_threads=4,
        display_progress=True,
        display_table=5,
    )
    score = evaluator(program)
    print(f"\nScore ({metric_name} metric): {score}")
    return score


if __name__ == "__main__":
    import sys

    metric_name = sys.argv[1] if len(sys.argv) > 1 else "keyword"
    run_eval(metric_name)
