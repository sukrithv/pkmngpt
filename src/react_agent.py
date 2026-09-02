"""DSPy Harness for Broad Pokemon Questions"""

import dspy

from src import pokeapi
from src.dspy_eval import configure_dspy

MAX_ITERS = 5

TOOLS = [
    pokeapi.pokemon_info,
    pokeapi.species_info,
    pokeapi.evolution_chain_for,
    pokeapi.type_relations,
    pokeapi.ability_info,
    pokeapi.move_info,
    pokeapi.pokemon_for_type,
    pokeapi.moves_for_type,
    pokeapi.moves_for_pokemon,
]


class BroadPokemonQA(dspy.Signature):
    """Answer a Pokemon question that may require checking facts across
    several Pokemon, types, or moves rather than looking up a single named
    entity. Use the provided tools to look up facts as needed instead of
    answering from memory alone. Keep calling tools until you have enough
    information to answer, then finish.
    """

    question = dspy.InputField()
    answer = dspy.OutputField(
        desc="Factual answer grounded in tool lookups, naming the Pokemon/types/moves involved"
    )


def build_agent(compiled_path=None):
    """Construct a fresh dspy.ReAct agent for broad Pokemon questions.

    Returns a dspy.Module. Call configure_dspy() before invoking it.
    """

    agent = dspy.ReAct(BroadPokemonQA, tools=TOOLS, max_iters=MAX_ITERS)
    if compiled_path is not None:
        agent.load(compiled_path)
    return agent


def run_broad_pokemon_qa(question):
    configure_dspy()
    agent = build_agent()
    prediction = agent(question=question)
    return {
        "question": question,
        "answer": prediction.answer,
        "trajectory": prediction.trajectory,
    }


def print_trajectory(trajectory):
    idx = 0
    while f"thought_{idx}" in trajectory:
        print(f"Thought {idx}: {trajectory[f'thought_{idx}']}")
        print(
            f"  Tool: {trajectory[f'tool_name_{idx}']}({trajectory[f'tool_args_{idx}']})"
        )
        print(f"  Observation: {trajectory[f'observation_{idx}']}")
        idx += 1


if __name__ == "__main__":
    import sys

    q = " ".join(sys.argv[1:]) or "Which fire types learn flying moves?"
    result = run_broad_pokemon_qa(q)
    print(f"Q: {result['question']}\n")
    print("Trajectory:")
    print_trajectory(result["trajectory"])
    print(f"\nA: {result['answer']}\n")
