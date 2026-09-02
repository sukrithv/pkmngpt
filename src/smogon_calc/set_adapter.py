"""Phase B — input resolution for the damage calculator.

Bridges the Smogon set store (get_sets) to the calc client's input shape.
Given a Pokemon (and optional gen/tier/set name), pull its competitive set and
convert it into the {species, item, ability, nature, evs, ...} dict that
calc_damage expects. Handles the "value or list-of-alternatives" set shape by
choosing one option (first by default, or caller-specified).
"""

from src.rag_llm.smogon_fetch import get_sets, load_set_store  # your set store
from src.smogon_calc.client import calc_damage  # your working client


def _pick(value, choice=0):
    """A set field may be a single value or a list of alternatives.
    Return one concrete value (the choice-th option, default first)."""
    if isinstance(value, list):
        if not value:
            return None
        return value[min(choice, len(value) - 1)]
    return value


def _pick_evs(evs, choice=0):
    """evs may be a single spread {'atk':252,...} or a list of alternative spreads."""
    if isinstance(evs, list):
        if not evs:
            return {}
        return evs[min(choice, len(evs) - 1)]
    return evs or {}


def _pick_moves(moves, move_choice=None):
    """Flatten a moveset's slots. Each slot is a move name or a list of options.
    Returns the list of concrete move names (one per slot). If move_choice is a
    dict {slot_index: option_index}, use it to pick specific options."""
    resolved = []
    for i, slot in enumerate(moves):
        if isinstance(slot, list):
            pick = 0
            if move_choice and i in move_choice:
                pick = move_choice[i]
            resolved.append(slot[min(pick, len(slot) - 1)])
        else:
            resolved.append(slot)
    return resolved


def set_to_calc_input(set_record: dict, choice: int = 0) -> dict:
    """Convert one Smogon set-store record into the calc client's Pokemon dict.
    `choice` selects among alternatives for item/nature/evs/tera when a set
    offers several (0 = the primary/first option)."""
    ivs = set_record.get("ivs") or {}
    calc_input = {
        "species": set_record["pokemon"].replace("-", " ").title().replace(" ", "-")
        if False
        else set_record["pokemon"],  # keep the slug; calc resolves it
        "ability": _pick(set_record.get("ability"), choice),
        "item": _pick(set_record.get("item"), choice),
        "nature": _pick(set_record.get("nature"), choice),
        "evs": _pick_evs(set_record.get("evs"), choice),
        "ivs": ivs,
        "teraType": _pick(set_record.get("teratypes"), choice),
    }
    if set_record.get("level") is not None:
        calc_input["level"] = set_record["level"]
    # drop None values so the calc uses its own defaults for anything unset
    return {k: v for k, v in calc_input.items() if v is not None and v != {}}


def resolve_set(store, pokemon, generation=None, tier=None, set_name=None):
    """Pull a single set record for a Pokemon from the store.
    If set_name is given, return that set; otherwise return the first available."""

    sets = get_sets(store, pokemon, generation=generation, tier=tier)

    if not sets:
        return None
    if set_name:
        for key, rec in sets.items():
            if rec["set_name"].lower() == set_name.lower():
                return rec
        return None  # named set not found

    sorted_sets = sorted(sets.items(), key=lambda x: x[1]["generation"], reverse=True)

    return sorted_sets[0][1]


def calc_with_sets(
    attacker_pokemon,
    defender_pokemon,
    move,
    generation=None,
    tier=None,
    attacker_set=None,
    defender_set=None,
    field=None,
    store=None,
):
    """High-level: compute damage using the two Pokemon's competitive sets from
    the store. `attacker_set`/`defender_set` optionally name a specific set."""
    store = store or load_set_store()

    atk_rec = resolve_set(store, attacker_pokemon, generation, tier, attacker_set)
    if atk_rec is None:
        raise ValueError(
            f"No {generation}/{tier} set found for attacker {attacker_pokemon}"
        )
    def_rec = resolve_set(store, defender_pokemon, generation, tier, defender_set)
    if def_rec is None:
        raise ValueError(
            f"No {generation}/{tier} set found for defender {defender_pokemon}"
        )

    gen_num = int(generation.replace("gen", "")) if generation else 9

    attacker = set_to_calc_input(atk_rec)
    defender = set_to_calc_input(def_rec)
    # attacker keeps its full set (item/ability matter); defender too.
    return calc_damage(attacker, defender, move, generation=gen_num, field=field or {})


if __name__ == "__main__":
    store = load_set_store()
    # compute using each Pokemon's actual competitive set
    r = calc_with_sets("zacian", "mr. rime", "Earthquake")
    print(r["desc"])
    print(f"{r['min']}-{r['max']}  {r.get('koText', '')}")
