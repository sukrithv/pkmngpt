import json
import random
import re
from pathlib import Path

import dspy

import config
from src.dspy_eval import configure_dspy, load_devset
from src.sql_llm.sql_agent import TextToSQL
from src.sql_llm.sql_pokeapi import run_query

SIMPLE_EVAL_SET_PATH = config.ROOT_DIR / "eval" / "simple_eval_set.json"
COMPLEX_EVAL_SET_PATH = config.ROOT_DIR / "eval" / "complex_eval_set.json"
TRAIN_VAL_SPLIT = 0.6
VAL_TEST_SPLIT = 0.8

COMPILED_REACT_AGENT_PATH = config.ROOT_DIR / "compiled_agent.json"


def load_devset(
    paths: list[Path] = [SIMPLE_EVAL_SET_PATH, COMPLEX_EVAL_SET_PATH],
) -> list[list[dspy.Example]]:
    examples = []
    for path in paths:
        if not path.exists():
            raise FileNotFoundError(f"{path} not found")
        raw = json.loads(path.read_text(encoding="utf-8"))
        subexamples = []
        for item in raw:
            example = dspy.Example(
                question=item["question"],
                expected_answer=item.get("expected_answer", ""),
                keywords=item.get("keywords", []),
                gold_answer=item.get("gold_answer", []),
                gold_sql=item.get("gold_sql", ""),
            ).with_inputs("question")
            subexamples.append(example)
        examples.append(subexamples)

    return examples


_GEN_ALIASES = {
    "1": "generation-i",
    "2": "generation-ii",
    "3": "generation-iii",
    "4": "generation-iv",
    "5": "generation-v",
    "6": "generation-vi",
    "7": "generation-vii",
    "8": "generation-viii",
    "9": "generation-ix",
    "generation-i": "generation-i",
    "generation-ii": "generation-ii",
    "generation-iii": "generation-iii",
    "generation-iv": "generation-iv",
    "generation-v": "generation-v",
    "generation-vi": "generation-vi",
    "generation-vii": "generation-vii",
    "generation-viii": "generation-viii",
    "generation-ix": "generation-ix",
}


def _norm_cell(v) -> str:
    s = str(v).strip().lower()
    s = re.sub(r"[,\s]", "", s)
    if re.fullmatch(r"-?\d+\.\d+", s):
        s = s.rstrip("0").rstrip(".")
    s = _GEN_ALIASES.get(s, s)
    return s


def _result_set(rows) -> set:
    return {_norm_cell(c) for row in rows for c in row}


def sql_result_metric(example, prediction, trace=None) -> bool:
    """Execute the model's SQL and compare its result to the gold answer.

    Passes if the model's query returns the same set of values as the gold
    query did. Requires:
      - example.gold_answer : the structured gold result (list of rows)
      - prediction.sql      : the SQL the agent generated
    """

    gold_rows = getattr(example, "gold_answer", None)
    model_sql = getattr(prediction, "sql", None)
    if gold_rows is None or not model_sql:
        return False

    try:
        _cols, model_rows = run_query(model_sql)
    except Exception:
        return False

    gold = _result_set(gold_rows)
    model = _result_set(model_rows)

    if not gold:
        return False

    if not gold.issubset(model):
        return False

    if len(model) > len(gold) + 2:
        return False

    return True


def optimize():
    configure_dspy()
    dataset = load_devset(paths=[SIMPLE_EVAL_SET_PATH, COMPLEX_EVAL_SET_PATH])
    train_set, val_set, test_set = [], [], []

    random.seed(config.SEED)
    for data in dataset:
        shuffled_data = random.sample(data, k=len(data))
        val_split_idx = round(len(shuffled_data) * TRAIN_VAL_SPLIT)
        test_split_idx = round(len(shuffled_data) * VAL_TEST_SPLIT)
        train_set.extend(shuffled_data[:val_split_idx])
        val_set.extend(shuffled_data[val_split_idx:test_split_idx])
        test_set.extend(shuffled_data[test_split_idx:])

    random.shuffle(train_set)
    random.shuffle(test_set)
    random.shuffle(val_set)

    agent = TextToSQL()
    optimizer = dspy.BootstrapFewShotWithRandomSearch(
        metric=sql_result_metric,
        max_bootstrapped_demos=4,
        max_labeled_demos=6,
        num_candidate_programs=4,
        num_threads=16,
        max_rounds=1,
    )

    compiled_agent = optimizer.compile(
        student=agent, trainset=train_set, valset=val_set
    )
    compiled_agent.save(COMPILED_REACT_AGENT_PATH, save_program=False)

    evaluator = dspy.evaluate.Evaluate(
        devset=test_set,
        metric=sql_result_metric,
        num_threads=8,
        display_progress=True,
        display_table=10,
    )

    score = evaluator(compiled_agent)
    print(f"\nScore: {score}")
    return score


if __name__ == "__main__":
    optimize()
