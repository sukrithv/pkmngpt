"""Routing eval: measure how accurately PokemonRouter's classifier picks the
right engine(s) for each question.

Reports:
  - per-flag accuracy (sql / pokemon_rag / concept_rag / api_tool)
  - exact-match accuracy (all flags correct at once)
  - pokemon_name extraction accuracy
  - a list of every misrouted question so you can see the pattern

Usage:  uv run python -m src.routing_eval
"""

import json
from collections import defaultdict
from pathlib import Path

from tqdm import tqdm

import config
from src.dspy_eval import configure_dspy
from src.router import PokemonRouter  # adjust import to your path

EVAL_PATH = config.ROOT_DIR / "eval" / "routing_eval_set.json"

FLAGS = ["needs_sql", "needs_pokemon_rag", "needs_concept_rag", "needs_api_tool"]


def _norm_name(v) -> str:
    s = (str(v) if v is not None else "none").strip().lower()
    return "none" if s in ("", "none", "null") else s


def run_routing_eval(router, examples):
    # per-flag tallies
    flag_correct = {f: 0 for f in FLAGS}
    flag_total = {f: 0 for f in FLAGS}
    exact_correct = 0
    name_correct = 0
    misroutes = []  # (question, expected_flags, got_flags)
    # confusion: for each flag, count false-pos and false-neg
    false_pos = defaultdict(int)
    false_neg = defaultdict(int)

    for ex in tqdm(examples, desc="ROUTING", unit="q"):
        q = ex["question"]
        # run only the router's CLASSIFIER, not the full gather+synthesize,
        # so we measure routing in isolation (fast, no SQL/RAG calls).
        d = router.classify(question=q)

        got = {f: bool(getattr(d, f)) for f in FLAGS}
        exp = {f: bool(ex[f]) for f in FLAGS}

        # per-flag
        all_match = True
        for f in FLAGS:
            flag_total[f] += 1
            if got[f] == exp[f]:
                flag_correct[f] += 1
            else:
                all_match = False
                if got[f] and not exp[f]:
                    false_pos[f] += 1  # turned on when it shouldn't
                if not got[f] and exp[f]:
                    false_neg[f] += 1  # missed when it should have

        if all_match:
            exact_correct += 1
        else:
            exp_on = [f.replace("needs_", "") for f in FLAGS if exp[f]]
            got_on = [f.replace("needs_", "") for f in FLAGS if got[f]]
            misroutes.append((q, exp_on, got_on))

        # name extraction
        if _norm_name(getattr(d, "pokemon_name", "none")) == _norm_name(
            ex["pokemon_name"]
        ):
            name_correct += 1

    n = len(examples)
    print(f"\n{'=' * 60}\nROUTING EVAL — {n} questions\n{'=' * 60}")
    print("Per-flag accuracy:")
    for f in FLAGS:
        acc = flag_correct[f] / flag_total[f]
        print(f"  {f:<20} {acc:>6.0%}   (FP={false_pos[f]}, FN={false_neg[f]})")
    print(
        f"\nExact-match (all flags right):  {exact_correct}/{n}  ({exact_correct / n:.0%})"
    )
    print(
        f"pokemon_name extraction:        {name_correct}/{n}  ({name_correct / n:.0%})"
    )

    if misroutes:
        print(f"\n{'-' * 60}\nMISROUTED ({len(misroutes)}):")
        for q, exp_on, got_on in misroutes:
            print(f"\n  Q: {q}")
            print(f"     expected: {exp_on or ['none']}")
            print(f"     got:      {got_on or ['none']}")

    return {
        "exact": exact_correct / n,
        "per_flag": {f: flag_correct[f] / flag_total[f] for f in FLAGS},
        "name": name_correct / n,
        "misroutes": misroutes,
    }


def main():
    configure_dspy()
    router = PokemonRouter()
    examples = json.loads(Path(EVAL_PATH).read_text(encoding="utf-8"))
    run_routing_eval(router, examples)


if __name__ == "__main__":
    main()
