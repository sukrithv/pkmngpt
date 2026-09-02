"""Phase C + D — damage calculator extraction and router integration.

Phase C: parse a natural-language damage question into structured calc inputs.
Phase D: the router branch that extracts, computes (via calc_with_sets), and
formats the result as evidence.

Reuses:
  - calc_with_sets / calc_damage (the working calc bridge + set adapter)
  - resolve_pokemon (your name-resolution fallback with fuzzy matching)
  - _clean_field (the quote/None normalizer from the router)
"""

import dspy

from src.smogon_calc.client import DamageServiceError, calc_damage
from src.smogon_calc.set_adapter import (
    load_set_store,
    resolve_set,
    set_to_calc_input,
)


# ------------------------------------------------------------------ Phase C
class ExtractDamageScenario(dspy.Signature):
    """Parse a Pokemon damage question into calculator inputs. Extract ONLY what
    the question states; leave a field as 'none' if unspecified (the calc layer
    fills competitive defaults). Weather examples: 'sun','rain','sand','snow'."""

    question: str = dspy.InputField()
    attacker: str = dspy.OutputField(desc="attacking Pokemon, lowercase slug")
    defender: str = dspy.OutputField(desc="defending Pokemon, lowercase slug")
    move: str = dspy.OutputField(desc="the attacking move")
    generation: str = dspy.OutputField(desc="e.g. 'gen9'; 'gen9' if unspecified")
    tier: str = dspy.OutputField(desc="e.g. 'ou'; 'ou' if unspecified")
    weather: str = dspy.OutputField(desc="weather if mentioned, else 'none'")
    attacker_item: str = dspy.OutputField(desc="attacker's item if stated, else 'none'")
    attacker_boost: str = dspy.OutputField(
        desc="attacker stat boost like '+1 atk' if stated, else 'none'"
    )


# weather phrase -> @smogon/calc field value
_WEATHER_MAP = {
    "sun": "Sun",
    "harsh sunlight": "Sun",
    "sunny": "Sun",
    "rain": "Rain",
    "raining": "Rain",
    "sand": "Sand",
    "sandstorm": "Sand",
    "snow": "Snow",
    "hail": "Hail",
}


def _clean_field(v):
    """Strip stray quotes/whitespace; treat 'none'/'null'/'' as empty."""
    if v is None:
        return ""
    s = str(v).strip().strip("'\"").strip().lower()
    return "" if s in ("none", "null", "") else s


def _parse_boost(text):
    """'+1 atk' -> {'atk': 1}, '+2 spa' -> {'spa': 2}. Returns {} if unparseable."""
    import re

    m = re.match(r"([+-]?\d+)\s*(hp|atk|def|spa|spd|spe)", text.strip().lower())
    if m:
        return {m.group(2): int(m.group(1))}
    return {}


# ------------------------------------------------------------------ Phase D
class DamageCalcModule(dspy.Module):
    """Router branch: NL damage question -> structured scenario -> calc -> evidence."""

    def __init__(self, resolve_pokemon_fn, known_names, set_store=None):
        self.extract = dspy.ChainOfThought(ExtractDamageScenario)
        self.resolve_pokemon = resolve_pokemon_fn  # your fuzzy name resolver
        self.known_names = known_names
        self.store = set_store or load_set_store()

    def forward(self, question: str) -> dspy.Prediction:
        s = self.extract(question=question)

        # normalize + resolve names (reuse your fuzzy fallback)
        attacker = self.resolve_pokemon(
            _clean_field(s.attacker), question, self.known_names
        )
        defender = self.resolve_pokemon(
            _clean_field(s.defender), question, self.known_names
        )
        move = _clean_field(s.move)
        gen = _clean_field(s.generation) or "gen9"
        tier = _clean_field(s.tier) or "ou"

        if not (attacker and defender and move):
            return dspy.Prediction(
                answer="I couldn't identify the attacker, defender, and move for a damage calc.",
                evidence="",
                ok=False,
            )

        # optional overrides from the question
        field = {}
        weather = _clean_field(s.weather)
        if weather in _WEATHER_MAP:
            field["weather"] = _WEATHER_MAP[weather]

        atk_override = {}
        item = _clean_field(s.attacker_item)
        if item:
            atk_override["item"] = item.title()  # 'choice band' -> 'Choice Band'
        boost = _parse_boost(_clean_field(s.attacker_boost))
        if boost:
            atk_override["boosts"] = boost

        # compute — try the store set first, fall back to bare species if no set
        try:
            result = self._compute(
                attacker, defender, move, gen, tier, field, atk_override
            )
        except (ValueError, DamageServiceError) as e:
            return dspy.Prediction(
                answer=f"Couldn't compute that damage: {e}", evidence="", ok=False
            )

        desc = result.get("desc", "")
        return dspy.Prediction(answer=desc, evidence=desc, result=result, ok=True)

    def _compute(self, attacker, defender, move, gen, tier, field, atk_override):
        """Use competitive sets; merge in any attacker overrides. Fall back to a
        bare-species calc if the store has no set for one of them."""
        gen_num = int(gen.replace("gen", ""))
        store = self.store

        atk_rec = resolve_set(store, attacker, gen, tier)
        def_rec = resolve_set(store, defender, gen, tier)

        # build attacker input: store set (if any) + overrides, else bare species
        if atk_rec:
            atk_in = set_to_calc_input(atk_rec)
        else:
            atk_in = {"species": attacker}
        atk_in.update(atk_override)  # question-stated item/boosts win

        def_in = set_to_calc_input(def_rec) if def_rec else {"species": defender}

        return calc_damage(atk_in, def_in, move, generation=gen_num, field=field or {})


# ------------------------------------------------------------------ Router demos
# add these to your ROUTER_DEMOS (with the other flags = False, and gen/tier)
DAMAGE_ROUTER_DEMOS_HINT = """
Add demos like:
  "How much does Choice Band Garchomp's Earthquake do to Kingambit?" -> needs_damage_calc=True
  "How much damage does Dragapult's Shadow Ball deal to Corviknight in sand?" -> needs_damage_calc=True
  "Does Garchomp OHKO Kingambit with Earthquake?" -> needs_damage_calc=True
Contrast (NOT damage calc):
  "What moves does competitive Garchomp run?" -> needs_smogon_sets=True (not damage_calc)
  "Why is Garchomp good in OU?" -> needs_smogon_analyses=True (not damage_calc)
"""
