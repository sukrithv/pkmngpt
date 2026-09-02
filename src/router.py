import json
import re

import dspy
from rapidfuzz import fuzz, process

import config
from src.rag_llm.rag import RAGAnswer
from src.rag_llm.smogon_fetch import format_sets, get_sets, load_set_store
from src.router_demos import ROUTER_DEMOS
from src.smogon_calc.calc_llm import DamageCalcModule
from src.sql_llm.sql_agent import TextToSQL

KNOWN_PKMN_NAMES = json.loads(
    (config.ROOT_DIR / "data" / "full_article_list.json").read_text(encoding="utf-8")
)["pokemon"]


class RouterClassifier(dspy.Signature):
    """Decide which knowledge sources are needed to answer a Pokemon question.
    - needs_sql: facts, stats, counts, comparisons ("which/how many/highest base X").
    - needs_pokemon_rag: a specific Pokemon's biology, behavior, or lore ("why is X violent").
    - needs_concept_rag: a game mechanic or concept ("what is Mega Evolution", "how does breeding work").
    - needs_api_tool: a specific move or ability's effect ("what does Flamethrower do").
    - needs_smogon_sets: a specific competitive moveset/build ("what's the standard X set")
    - needs_smogon_analyses: competitive reasoning/viability ("why is X good in OU", "what checks X")
    - needs_damage_calc: a damage calculation from one pokemon's move on another (how much does X's move do to Y")
    - generation: the specific generation the question is about, in formats like 'gen9', or None
    - tier: the specific competitive tier the question is about, or None
    - pokemon_name: the specific Pokemon the question is about, lowercase slug, or None.
    Set every flag that applies — a question may need more than one source."""

    question: str = dspy.InputField()
    needs_sql: bool = (
        dspy.OutputField()
    )  # facts, stats, counts, "which/how many/highest"
    needs_pokemon_rag: bool = (
        dspy.OutputField()
    )  # a specific Pokemon's biology/behavior/lore
    needs_concept_rag: bool = (
        dspy.OutputField()
    )  # a mechanic/concept ("what is Mega Evolution")
    needs_api_tool: bool = dspy.OutputField()  # a specific move/ability effect
    needs_smogon_sets: bool = (
        dspy.OutputField()
    )  # a specific competitive moveset/build ("what's the standard X set")
    needs_smogon_analyses: bool = (
        dspy.OutputField()
    )  # competitive reasoning/viability ("why is X good in OU", "what checks X")
    needs_damage_calc: bool = dspy.OutputField()
    generation: str | None = dspy.OutputField(
        desc="Smogon generation like 'gen1','gen2',...,'gen9','gen9champions'"
    )
    tier: str | None = dspy.OutputField(
        desc="Smogon tier like 'ubers','ou','uu','ru','nu','pu','zu', or None"
    )
    pokemon_name: str | None = dspy.OutputField(
        desc="lowercase Pokemon slug, or 'none'"
    )  # extracted subject, if any


class SynthesizeAnswer(dspy.Signature):
    """Answer the question using ONLY the provided evidence. Database facts are
    exact and authoritative; prose passages provide explanation and lore. If a
    fact and prose disagree, trust the database fact. If the evidence doesn't
    contain the answer, say so. Be concise."""

    question: str = dspy.InputField()
    context: str = dspy.InputField()
    answer: str = dspy.OutputField(desc="Gathered results, each labeled by source")


def _format_evidence(evidence):
    blocks = []
    if evidence.get("facts") is not None:
        blocks.append(f"[DATABASE FACTS]\n{evidence['facts']}")
    if evidence.get("pokemon_prose"):
        blocks.append(f"[POKEMON LORE]\n{evidence['pokemon_prose']}")
    if evidence.get("concept_prose"):
        blocks.append(f"[MECHANIC/CONCEPT]\n{evidence['concept_prose']}")
    if evidence.get("smogon_set"):
        blocks.append(f"[SMOGON SETS]\n{evidence['smogon_set']}")
    if evidence.get("smogon_analysis"):
        blocks.append(f"[SMOGON ANALYSES]\n{evidence['smogon_analysis']}")
    if evidence.get("tool_result"):
        blocks.append(f"[MOVE/ABILITY DATA]\n{evidence['tool_result']}")
    if evidence.get("damage"):
        blocks.append(f"[DAMAGE CALC]\n{evidence['damage']}")
    return "\n\n".join(blocks) if blocks else "(no evidence gathered)"


def _clean_field(v: str) -> str:
    """Strip stray quotes/whitespace the model sometimes wraps values in."""
    if v is None:
        return ""
    s = str(v).strip().strip("'\"").strip().lower()
    return "" if s in ("none", "null", "") else s


def _fuzzy_match_pokemon(candidate, known_names=KNOWN_PKMN_NAMES, cutoff=80):
    match = process.extractOne(
        candidate.lower(), known_names, scorer=fuzz.ratio, score_cutoff=cutoff
    )
    return match[0] if match else None


def _resolve_pokemon(candidate, question, known_names=KNOWN_PKMN_NAMES):
    if not candidate:
        candidate = ""
    c = candidate.lower().strip()

    # 1. exact match (model got it right)
    if c in known_names:
        return c
    # 2. substring in question (fallback: name present but classifier missed it)
    exact_in_q = [
        n for n in known_names if re.search(rf"\b{re.escape(n)}\b", question.lower())
    ]
    if exact_in_q:
        return max(exact_in_q, key=len)
    # 3. fuzzy match (typo in the candidate the model extracted)
    fuzzy = _fuzzy_match_pokemon(c, known_names, cutoff=0.8)
    if fuzzy:
        return fuzzy
    # 4. fuzzy against each word in the question (typo the model didn't catch)
    for word in re.findall(r"\b\w{4,}\b", question.lower()):
        fuzzy = _fuzzy_match_pokemon(word, known_names, cutoff=0.85)
        if fuzzy:
            return fuzzy
    return None


class PokemonRouter(dspy.Module):
    """Routes question to correct model"""

    def __init__(self):
        self.texttosql = TextToSQL()
        self.rag = RAGAnswer()
        self.classify = dspy.ChainOfThought(RouterClassifier)
        self.syntesize = dspy.ChainOfThought(SynthesizeAnswer)
        self.set_store = load_set_store()
        self.damage = DamageCalcModule(
            _resolve_pokemon, KNOWN_PKMN_NAMES, self.set_store
        )

        self.classify.predict.demos = ROUTER_DEMOS

    def forward(self, question: str) -> dspy.Prediction:
        decision = self.classify(question=question)
        name = _clean_field(decision.pokemon_name)
        name = _resolve_pokemon(name, question)

        gen = _clean_field(decision.generation) if decision.generation else "gen9"
        tier = _clean_field(decision.tier) if decision.tier else "ou"

        evidence = {}
        if decision.needs_sql:
            try:
                sql_pred = self.texttosql(question=question)
                evidence["facts"] = (
                    f"{sql_pred.answer}\n(query: {getattr(sql_pred, 'sql', '')})"
                )
            except Exception as e:
                evidence["facts"] = f"SQL Engine Error: {e}"

        if decision.needs_pokemon_rag:
            rag_pred = self.rag(question=question, pokemon_name=name)
            evidence["pokemon_prose"] = rag_pred.context

        if decision.needs_concept_rag:
            rag_pred = self.rag(question=question)
            evidence["concept_prose"] = rag_pred.context

        if decision.needs_smogon_sets:
            set_data = get_sets(
                store=self.set_store, pokemon=name, generation=gen, tier=tier
            )
            evidence["smogon_set"] = format_sets(set_data)

        if decision.needs_smogon_analyses:
            rag_pred = self.rag(
                question=question,
                pokemon_name=name,
                doc_type="smogon",
                generation=gen,
                tier=tier,
            )
            evidence["smogon_analysis"] = rag_pred.context

        if decision.needs_damage_calc:
            dmg_pred = self.damage(question=question)
            evidence["damage"] = dmg_pred.answer

        if not any(
            [
                decision.needs_sql,
                decision.needs_pokemon_rag,
                decision.needs_concept_rag,
                decision.needs_smogon_sets,
                decision.needs_smogon_analyses,
                decision.needs_damage_calc,
            ]
        ):
            rag_pred = self.rag(question=question, pokemon_name=name)
            evidence["pokemon_prose"] = rag_pred.context

        context = _format_evidence(evidence=evidence)
        final = self.syntesize(question=question, context=context)

        return dspy.Prediction(
            answer=final.answer,
            routing=decision,
            evidence=evidence,
        )


if __name__ == "__main__":
    from src.dspy_eval import configure_dspy

    configure_dspy()
    router = PokemonRouter()

    DAMAGE_TEST_QUESTIONS = [
        # --- 1. Core happy path: route to damage_calc, return a range ---
        {
            "q": "How much does Choice Band Garchomp's Earthquake do to Kingambit?",
            "expected_branch": "damage_calc",
            "note": "basic flow",
        },
        {
            "q": "How much damage does Dragapult's Shadow Ball deal to Great Tusk?",
            "expected_branch": "damage_calc",
            "note": "basic flow",
        },
        {
            "q": "What damage does Iron Valiant's Moonblast do to Kingambit?",
            "expected_branch": "damage_calc",
            "note": "basic flow",
        },
        # --- 2. KO-phrasing: same intent, different wording ---
        {
            "q": "Does Garchomp OHKO Kingambit with Earthquake?",
            "expected_branch": "damage_calc",
            "note": "OHKO phrasing",
        },
        {
            "q": "Will Dragapult's Draco Meteor 2HKO Corviknight?",
            "expected_branch": "damage_calc",
            "note": "2HKO phrasing",
        },
        {
            "q": "Can Iron Valiant knock out Great Tusk with Moonblast?",
            "expected_branch": "damage_calc",
            "note": "knock-out phrasing",
        },
        # --- 3. Conditions: weather / boosts / items must be extracted ---
        {
            "q": "How much does Garchomp's Earthquake do to Corviknight in the sand?",
            "expected_branch": "damage_calc",
            "note": "weather + IMMUNITY (ground vs flying)",
        },
        {
            "q": "How much does +2 Kingambit's Sucker Punch do to Gholdengo?",
            "expected_branch": "damage_calc",
            "note": "boost extraction",
        },
        {
            "q": "How much does Life Orb Dragapult's Shadow Ball do to Great Tusk?",
            "expected_branch": "damage_calc",
            "note": "item override",
        },
        {
            "q": "How much does Choice Specs Iron Valiant's Moonblast do to Kingambit in sun?",
            "expected_branch": "damage_calc",
            "note": "item + weather",
        },
        # --- 4. Immunity / no-effect: graceful zero-damage end to end ---
        {
            "q": "How much does Earthquake do to Corviknight?",
            "expected_branch": "damage_calc",
            "note": "type immunity (ground->flying)",
        },
        {
            "q": "How much does Sucker Punch do to Gholdengo?",
            "expected_branch": "damage_calc",
            "note": "ability immunity (Good as Gold)",
        },
        {
            "q": "How much does a Normal move do to a Ghost type?",
            "expected_branch": "damage_calc",
            "note": "generic type immunity",
        },
        # --- 5. Missing-set fallback: off-tier Pokemon -> bare-species or message ---
        {
            "q": "How much does Pikachu's Thunderbolt do to Snorlax?",
            "expected_branch": "damage_calc",
            "note": "off-tier attacker, no set -> fallback",
        },
        {
            "q": "How much does Magikarp's Splash do to Charizard?",
            "expected_branch": "damage_calc",
            "note": "off-tier + zero-power move",
        },
        # --- 6. Ambiguous / underspecified: must FAIL GRACEFULLY, not crash ---
        {
            "q": "How much damage does Garchomp do?",
            "expected_branch": "damage_calc",
            "note": "no defender/move -> graceful failure",
        },
        {
            "q": "How much does Earthquake do?",
            "expected_branch": "damage_calc",
            "note": "no attacker/defender -> graceful failure",
        },
        {
            "q": "Which move does the most damage?",
            "expected_branch": "unclear",
            "note": "no specifics -> graceful failure or clarify",
        },
        # --- 7. BOUNDARY: must NOT route to damage_calc (over-trigger test) ---
        {
            "q": "What's Garchomp's competitive set?",
            "expected_branch": "smogon_sets",
            "note": "NOT damage_calc",
        },
        {
            "q": "Why is Garchomp good in OU?",
            "expected_branch": "smogon_analyses",
            "note": "NOT damage_calc",
        },
        {
            "q": "What does Earthquake do?",
            "expected_branch": "api_tool",
            "note": "move effect, NOT damage_calc",
        },
        {
            "q": "What's Garchomp's base Attack?",
            "expected_branch": "sql",
            "note": "a stat, NOT damage_calc",
        },
        {
            "q": "Is Garchomp strong?",
            "expected_branch": "smogon_analyses",
            "note": "vague viability, NOT damage_calc",
        },
        # --- 8. Cross-generation: gen/tier extraction must feed the calc ---
        {
            "q": "How much does Gen 4 Garchomp's Outrage do to Skarmory?",
            "expected_branch": "damage_calc",
            "note": "extract gen4, gen4 mechanics",
        },
        {
            "q": "In Gen 1, how much does Tauros's Body Slam do to Chansey?",
            "expected_branch": "damage_calc",
            "note": "extract gen1, gen1 mechanics",
        },
    ]

    for case in DAMAGE_TEST_QUESTIONS:
        pred = router(question=case["q"])
        r = pred.routing
        fired = [
            f
            for f in [
                "sql",
                "pokemon_rag",
                "concept_rag",
                "api_tool",
                "smogon_sets",
                "smogon_analyses",
                "damage_calc",
            ]
            if getattr(r, f"needs_{f}", False)
        ]
        print(f"\nQ: {case['q']}")
        print(f"   expected: {case['expected_branch']}  | note: {case['note']}")
        print(f"   fired:    {fired}")
        print(f"   answer:   {pred.answer}")
