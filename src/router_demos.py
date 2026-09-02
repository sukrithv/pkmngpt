"""Complete ROUTER_DEMOS for PokemonRouter's classifier.

Every demo sets ALL output flags (so examples render completely) plus
pokemon_name / generation / tier. Covers every branch:
  sql, pokemon_rag, concept_rag, api_tool, smogon_sets, smogon_analyses,
  damage_calc — including multi-branch and contrast cases that teach the
  boundaries between similar-sounding intents.

Attach with:  self.classify.predict.demos = ROUTER_DEMOS
"""

import dspy


def _demo(
    question,
    reasoning,
    *,
    sql=False,
    pkmn=False,
    concept=False,
    api=False,
    sets=False,
    analyses=False,
    damage=False,
    name="none",
    gen="none",
    tier="none",
):
    return dspy.Example(
        question=question,
        reasoning=reasoning,
        needs_sql=sql,
        needs_pokemon_rag=pkmn,
        needs_concept_rag=concept,
        needs_api_tool=api,
        needs_smogon_sets=sets,
        needs_smogon_analyses=analyses,
        needs_damage_calc=damage,
        pokemon_name=name,
        generation=gen,
        tier=tier,
    ).with_inputs("question")


ROUTER_DEMOS = [
    # ---------------- SQL: facts, stats, counts, comparisons ----------------
    _demo(
        "Which Generation I Fire-type has the highest base Attack?",
        "A superlative stat comparison — pure database query. No lore or mechanic.",
        sql=True,
    ),
    _demo(
        "How many Water-type Pokemon are there in Generation III?",
        "A count over the database. SQL.",
        sql=True,
    ),
    _demo(
        "What egg group does Pikachu belong to?",
        "A specific egg-group fact for one Pokemon — database lookup, not lore.",
        sql=True,
        name="pikachu",
    ),
    _demo(
        "What does Bulbasaur evolve into?",
        "A specific evolution fact — the evolves-into relation is in the database.",
        sql=True,
        name="bulbasaur",
    ),
    _demo(
        "Which nature raises Attack and lowers Special Attack?",
        "A lookup of a specific nature's stat effects — exact data in the natures table.",
        sql=True,
    ),
    # ---------------- Pokemon RAG: creature biology / behavior / lore -------
    _demo(
        "Why is Gyarados so violent?",
        "Asks about a specific Pokemon's temperament/lore. Pokemon RAG.",
        pkmn=True,
        name="gyarados",
    ),
    _demo(
        "How does Gengar hide from its prey?",
        "Creature behavior/lore for a specific Pokemon. Pokemon RAG.",
        pkmn=True,
        name="gengar",
    ),
    _demo(
        "What is the origin and design inspiration behind Charizard?",
        "Lore/design prose for a specific Pokemon. Pokemon RAG.",
        pkmn=True,
        name="charizard",
    ),
    _demo(
        "What happens in the Sun and Moon story?",
        "Game plot/lore, not a specific creature fact. Pokemon/lore RAG.",
        pkmn=True,
    ),
    # ---------------- Concept RAG: mechanics / concepts ---------------------
    _demo(
        "What is Mega Evolution?",
        "Asks about a game mechanic/concept — concept RAG.",
        concept=True,
    ),
    _demo(
        "How does Pokemon breeding work?",
        "A mechanic explanation — concept RAG.",
        concept=True,
    ),
    _demo(
        "How does the Erratic experience group work?",
        "Explains an experience-curve concept — concept RAG.",
        concept=True,
    ),
    _demo(
        "How do abilities work in general?",
        "The ability MECHANIC as a concept, not one specific ability. Concept RAG, "
        "not api_tool.",
        concept=True,
    ),
    # ---------------- API tool: a specific move/ability effect --------------
    _demo(
        "What does the move Flamethrower do?",
        "A specific move's effect — direct move/ability lookup (api_tool).",
        api=True,
    ),
    _demo(
        "What is the effect of the Static ability?",
        "A specific ability's effect — api_tool lookup, not the general concept.",
        api=True,
    ),
    _demo(
        "Explain the ability Intimidate.",
        "A specific ability's effect — api_tool.",
        api=True,
    ),
    # ---------------- Smogon sets: a specific competitive build -------------
    _demo(
        "What's the standard competitive set for Garchomp in OU?",
        "Asks for a specific competitive moveset/build — Smogon sets lookup. "
        "Tier OU stated; generation defaults to gen9.",
        sets=True,
        name="garchomp",
        gen="gen9",
        tier="ou",
    ),
    _demo(
        "What moves does competitive Dragapult run?",
        "Asks for a competitive moveset — Smogon sets. No format stated, default gen9/ou.",
        sets=True,
        name="dragapult",
        gen="gen9",
        tier="ou",
    ),
    # ---------------- Smogon analyses: competitive reasoning ---------------
    _demo(
        "Why is Garchomp good in Gen 9 OU?",
        "Competitive viability/reasoning — Smogon analysis prose, not a moveset. "
        "Gen 9 OU stated.",
        analyses=True,
        name="garchomp",
        gen="gen9",
        tier="ou",
    ),
    _demo(
        "What checks and counters Dragapult?",
        "Competitive matchup reasoning — Smogon analysis. Default gen9/ou.",
        analyses=True,
        name="dragapult",
        gen="gen9",
        tier="ou",
    ),
    # ---------------- Damage calc: how much a move does --------------------
    _demo(
        "How much does Choice Band Garchomp's Earthquake do to Kingambit?",
        "Asks for a specific damage amount between two Pokemon — damage calculation.",
        damage=True,
        name="garchomp",
        gen="gen9",
        tier="ou",
    ),
    _demo(
        "Does Dragapult OHKO Corviknight with Shadow Ball?",
        "A KO/damage question between two Pokemon — damage calc.",
        damage=True,
        name="dragapult",
        gen="gen9",
        tier="ou",
    ),
    _demo(
        "How much damage does Iron Valiant's Moonblast deal to Great Tusk in the sun?",
        "A damage amount with conditions (weather) — damage calc.",
        damage=True,
        name="iron-valiant",
        gen="gen9",
        tier="ou",
    ),
    # ---------------- Multi-branch cases -----------------------------------
    _demo(
        "Give me a good competitive Kingambit and explain why it works.",
        "Needs the specific set (smogon_sets) AND the viability reasoning "
        "(smogon_analyses). Both. Default gen9/ou.",
        sets=True,
        analyses=True,
        name="kingambit",
        gen="gen9",
        tier="ou",
    ),
    # ---------------- Contrast cases (teach the boundaries) ----------------
    _demo(
        "Why is Gyarados so aggressive?",
        "Creature temperament/lore — pokemon_rag, NOT Smogon competitive analysis. "
        "'Aggressive' here is biology, not a battle role.",
        pkmn=True,
        name="gyarados",
    ),
    _demo(
        "What is a competitive Pokemon set?",
        "Asks about the CONCEPT of a competitive set, not a specific Pokemon's set. "
        "Concept RAG, not smogon_sets.",
        concept=True,
    ),
]
