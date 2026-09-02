


from src import pokeapi

CHARIZARD_SPECIES = {
    "genera": [{"language": {"name": "en"}, "genus": "Flame Pokémon"}],
    "flavor_text_entries": [
        {
            "language": {"name": "en"},
            "flavor_text": "Spits fire that\nis hot enough to melt boulders.",
        },
        {
            "language": {"name": "en"},
            "flavor_text": "Spits fire that is hot enough to melt boulders.",
        },
        {"language": {"name": "de"}, "flavor_text": "Speit Feuer."},
    ],
}

CHARIZARD_POKEMON = {
    "types": [
        {"slot": 2, "type": {"name": "flying"}},
        {"slot": 1, "type": {"name": "fire"}},
    ],
    "abilities": [
        {"ability": {"name": "blaze"}, "is_hidden": False},
        {"ability": {"name": "solar-power"}, "is_hidden": True},
    ],
    "stats": [{"stat": {"name": "hp"}, "base_stat": 78}],
    "moves": [
        {"move": {"name": "flamethrower"}},
        {"move": {"name": "flamethrower"}},
        {"move": {"name": "ember"}},
    ],
}

FIRE_TYPE = {
    "name": "fire",
    "damage_relations": {
        "double_damage_to": [{"name": "grass"}],
        "half_damage_to": [{"name": "fire"}],
        "no_damage_to": [],
        "double_damage_from": [{"name": "water"}],
        "half_damage_from": [{"name": "bug"}],
        "no_damage_from": [],
    },
}

LINEAR_CHAIN = {
    "chain": {
        "species": {"name": "charmander"},
        "evolves_to": [
            {
                "species": {"name": "charmeleon"},
                "evolution_details": [
                    {"trigger": {"name": "level-up"}, "min_level": 16}
                ],
                "evolves_to": [
                    {
                        "species": {"name": "charizard"},
                        "evolution_details": [
                            {"trigger": {"name": "level-up"}, "min_level": 36}
                        ],
                        "evolves_to": [],
                    }
                ],
            }
        ],
    }
}

BRANCHING_CHAIN = {
    "chain": {
        "species": {"name": "eevee"},
        "evolves_to": [
            {
                "species": {"name": "vaporeon"},
                "evolution_details": [
                    {"trigger": {"name": "use-item"}, "item": {"name": "water-stone"}}
                ],
                "evolves_to": [],
            },
            {
                "species": {"name": "jolteon"},
                "evolution_details": [
                    {"trigger": {"name": "use-item"}, "item": {"name": "thunder-stone"}}
                ],
                "evolves_to": [],
            },
            {
                "species": {"name": "flareon"},
                "evolution_details": [
                    {"trigger": {"name": "use-item"}, "item": {"name": "fire-stone"}}
                ],
                "evolves_to": [],
            },
        ],
    }
}

NO_EVOLUTION_CHAIN = {"chain": {"species": {"name": "tauros"}, "evolves_to": []}}

SWORDS_DANCE = {
    "name": "swords-dance",
    "type": {"name": "normal"},
    "damage_class": {"name": "status"},
    "power": None,
    "accuracy": None,
    "pp": 20,
    "effect_entries": [
        {"language": {"name": "en"}, "effect": "Sharply raises the user's Attack stat."}
    ],
}

BLAZE_ABILITY = {
    "effect_entries": [
        {
            "language": {"name": "en"},
            "effect": "Powers up Fire-type moves when the Pokemon's HP is low.",
        },
    ],
}

FIRE_TYPE_WITH_POKEMON_AND_MOVES = {
    "pokemon": [
        {"slot": 1, "pokemon": {"name": "charizard"}},
        {"slot": 1, "pokemon": {"name": "charmander"}},
        {"slot": 1, "pokemon": {"name": "cyndaquil"}},  # gen-2, not in GEN_1_NAMES
    ],
    "moves": [
        {"name": "flamethrower"},
        {"name": "fire-blast"},
        {"name": "not-a-real-move"},  # not in MOVE_NAMES
    ],
}

CHARIZARD_POKEMON_WITH_VERSION_DETAILS = {
    "moves": [
        {
            "move": {"name": "flamethrower"},
            "version_group_details": [
                {"version_group": {"name": "red-blue"}},
                {"version_group": {"name": "gold-silver"}},
            ],
        },
        {
            "move": {"name": "dragon-claw"},
            "version_group_details": [
                {"version_group": {"name": "ruby-sapphire"}},
            ],
        },
        {
            "move": {"name": "growl"},
            "version_group_details": [
                {"version_group": {"name": "yellow"}},
            ],
        },
        {
            "move": {"name": "scratch"},
            "version_group_details": [
                {"version_group": {"name": "red-blue"}},
                {"version_group": {"name": "yellow"}},
            ],
        },
    ],
}


class TestMatch:
    def test_match_finds_name(self):
        assert pokeapi._match(["mr-mime"], "what does mr-mime evolve from?") == [
            "mr-mime"
        ]
        assert pokeapi._match(["mr-mime"], "what does mr mime evolve from?") == [
            "mr-mime"
        ]

    def test_match_false_positive(self):
        assert "ice" not in pokeapi._match(
            pokeapi.TYPE_NAMES, "do you have any advice for me?"
        )
        assert "rest" not in pokeapi._match(
            pokeapi.MOVE_NAMES, "which pokemon learns forest curse?"
        )


class TestFormat:
    def test_format_species_dedupes_flavor_text(self):
        text = pokeapi._format_species(CHARIZARD_SPECIES)
        assert text.count("Spits fire") == 1
        assert "Speit Feuer" not in text

    def test_format_pokemon_sorts_correctly(self):
        assert "Types: fire, flying" in pokeapi._format_pokemon(CHARIZARD_POKEMON)
        assert "ember, flamethrower" in pokeapi._format_pokemon(CHARIZARD_POKEMON)

    def test_format_pokemon_marks_hidden_ability(self):
        text = pokeapi._format_pokemon(CHARIZARD_POKEMON)
        assert "solar-power (hidden)" in text
        assert "blaze (hidden)" not in text

    def test_evolution_triggers(self):
        details = [{"trigger": {"name": "level-up"}, "min_level": 16}]
        assert pokeapi._describe_evolution_trigger(details) == "at level 16"

        details = [{"trigger": {"name": "use-item"}, "item": {"name": "fire-stone"}}]
        assert pokeapi._describe_evolution_trigger(details) == "using a fire-stone"

        details = [{"trigger": {"name": "trade"}, "item": None}]
        assert pokeapi._describe_evolution_trigger(details) == "by trading"

        assert pokeapi._describe_evolution_trigger([]) == "somehow"

        details = [{"trigger": {"name": "level-up-happiness"}}]
        assert pokeapi._describe_evolution_trigger(details) == "level up happiness"

    def test_format_evolution_chain(self):
        assert (
            pokeapi._format_evolution_chain(NO_EVOLUTION_CHAIN)
            == "This Pokemon does not evolve."
        )

        text = pokeapi._format_evolution_chain(LINEAR_CHAIN)
        assert "charmander evolves into charmeleon at level 16." in text
        assert "charmeleon evolves into charizard at level 36." in text

        text = pokeapi._format_evolution_chain(BRANCHING_CHAIN)
        assert "eevee evolves into vaporeon using a water-stone." in text
        assert "eevee evolves into jolteon using a thunder-stone." in text
        assert "eevee evolves into flareon using a fire-stone." in text

    def test_format_type_lists_all_relations(self):
        text = pokeapi._format_type(FIRE_TYPE)
        assert "deals double damage to: grass" in text
        assert "Takes no damage from: none" in text
        assert text.startswith("Fire type")

    def test_format_move_handles_status_move_with_no_power_or_accuracy(self):
        text = pokeapi._format_move(SWORDS_DANCE)
        assert "Power None" in text
        assert "accuracy None" in text
        assert "Swords dance is a normal-type status move" in text


class TestQuery:
    def test_query_multiple_resource_types(self, monkeypatch):
        def fake_fetch(url):
            if "/pokemon-species/charizard" in url:
                return CHARIZARD_SPECIES
            if "/pokemon/charizard" in url:
                return CHARIZARD_POKEMON
            if "/type/fire" in url:
                return FIRE_TYPE
            return None

        monkeypatch.setattr(pokeapi, "_fetch", fake_fetch)
        hits = pokeapi.query("is charizard's fire type strong against grass?")
        sources = {h["source"] for h in hits}
        assert {
            "pokeapi:species:charizard",
            "pokeapi:pokemon:charizard",
            "pokeapi:type:fire",
        } <= sources

    def test_query_evol_keyword_triggers_chain_fetch(self, monkeypatch):
        def fake_fetch(url):
            if "pokemon-species/charizard" in url:
                return {
                    **CHARIZARD_SPECIES,
                    "evolution_chain": {
                        "url": "https://pokeapi.co/api/v2/evolution-chain/2/"
                    },
                }
            if "/pokemon/charizard" in url:
                return CHARIZARD_POKEMON
            if "evolution-chain" in url:
                return LINEAR_CHAIN
            return None

        monkeypatch.setattr(pokeapi, "_fetch", fake_fetch)
        hits = pokeapi.query("what does charizard evolve from?")
        assert any(h["source"] == "pokeapi:evolution-chain:charizard" for h in hits)

    def test_query_evol_false_positive_substring(self, monkeypatch):
        # "revolution" contains "evol" -- an unrelated question shouldn't trigger
        # a chain fetch, but the current substring check does.
        calls = []

        def fake_fetch(url):
            calls.append(url)
            if "pokemon-species/charizard" in url:
                return {
                    **CHARIZARD_SPECIES,
                    "evolution_chain": {
                        "url": "https://pokeapi.co/api/v2/evolution-chain/2/"
                    },
                }
            if "/pokemon/charizard" in url:
                return CHARIZARD_POKEMON
            return None

        monkeypatch.setattr(pokeapi, "_fetch", fake_fetch)
        pokeapi.query("was charizard part of any revolution in pokemon design?")
        assert not any(
            "evolution-chain" in u for u in calls
        )  # currently FAILS -- pins the bug

    def test_query_top_k_caps_matches_per_resource_list(self, monkeypatch):
        monkeypatch.setattr(pokeapi, "_fetch_species", lambda name: None)
        monkeypatch.setattr(pokeapi, "_fetch_pokemon", lambda name: CHARIZARD_POKEMON)
        hits = pokeapi.query("charizard blastoise venusaur charmander", top_k=2)
        pokemon_hits = [h for h in hits if h["source"].startswith("pokeapi:pokemon:")]
        assert len(pokemon_hits) == 2

    def test_query_partial_failure_still_returns_successful_half(self, monkeypatch):
        def fake_fetch(url):
            if "pokemon-species/charizard" in url:
                return None  # simulated 404
            if "/pokemon/charizard" in url:
                return CHARIZARD_POKEMON
            return None

        monkeypatch.setattr(pokeapi, "_fetch", fake_fetch)
        hits = pokeapi.query("charizard")
        assert any(h["source"] == "pokeapi:pokemon:charizard" for h in hits)
        assert not any(h["source"] == "pokeapi:species:charizard" for h in hits)


class TestDSPYWrappers:
    def test_pokemon_info_found(self, monkeypatch):
        monkeypatch.setattr(pokeapi, "_fetch_pokemon", lambda name: CHARIZARD_POKEMON)
        text = pokeapi.pokemon_info("charizard")
        assert text == pokeapi._format_pokemon(CHARIZARD_POKEMON)

    def test_pokemon_info_not_found(self, monkeypatch):
        monkeypatch.setattr(pokeapi, "_fetch_pokemon", lambda name: None)
        assert (
            pokeapi.pokemon_info("not-a-real-mon")
            == "No Pokemon named 'not-a-real-mon' found."
        )

    def test_species_info_found(self, monkeypatch):
        monkeypatch.setattr(pokeapi, "_fetch_species", lambda name: CHARIZARD_SPECIES)
        text = pokeapi.species_info("charizard")
        assert text == pokeapi._format_species(CHARIZARD_SPECIES)

    def test_species_info_not_found(self, monkeypatch):
        monkeypatch.setattr(pokeapi, "_fetch_species", lambda name: None)
        assert (
            pokeapi.species_info("not-a-real-mon")
            == "No Pokemon species named 'not-a-real-mon' found."
        )

    def test_evolution_chain_for_found(self, monkeypatch):
        species_with_chain = {
            **CHARIZARD_SPECIES,
            "evolution_chain": {"url": "https://pokeapi.co/api/v2/evolution-chain/2/"},
        }
        monkeypatch.setattr(pokeapi, "_fetch_species", lambda name: species_with_chain)
        monkeypatch.setattr(pokeapi, "_fetch_evolution_chain", lambda url: LINEAR_CHAIN)
        text = pokeapi.evolution_chain_for("charmander")
        assert "charmander evolves into charmeleon at level 16." in text

    def test_evolution_chain_for_species_not_found(self, monkeypatch):
        monkeypatch.setattr(pokeapi, "_fetch_species", lambda name: None)
        assert (
            pokeapi.evolution_chain_for("not-a-real-mon")
            == "No Pokemon species named 'not-a-real-mon' found."
        )

    def test_evolution_chain_for_missing_evolution_chain_field(self, monkeypatch):
        monkeypatch.setattr(
            pokeapi, "_fetch_species", lambda name: {**CHARIZARD_SPECIES}
        )
        assert (
            pokeapi.evolution_chain_for("charizard")
            == "'charizard' has no evolution chain data."
        )

    def test_evolution_chain_for_chain_fetch_fails(self, monkeypatch):
        species_with_chain = {
            **CHARIZARD_SPECIES,
            "evolution_chain": {"url": "https://pokeapi.co/api/v2/evolution-chain/2/"},
        }
        monkeypatch.setattr(pokeapi, "_fetch_species", lambda name: species_with_chain)
        monkeypatch.setattr(pokeapi, "_fetch_evolution_chain", lambda url: None)
        assert (
            pokeapi.evolution_chain_for("charizard")
            == "Could not fetch the evolution chain for 'charizard'."
        )

    def test_type_relations_found(self, monkeypatch):
        monkeypatch.setattr(pokeapi, "_fetch_type", lambda name: FIRE_TYPE)
        text = pokeapi.type_relations("fire")
        assert text == pokeapi._format_type(FIRE_TYPE)

    def test_type_relations_not_found(self, monkeypatch):
        monkeypatch.setattr(pokeapi, "_fetch_type", lambda name: None)
        assert (
            pokeapi.type_relations("not-a-real-type")
            == "No type named 'not-a-real-type' found."
        )

    def test_ability_info_found(self, monkeypatch):
        monkeypatch.setattr(pokeapi, "_fetch_ability", lambda name: BLAZE_ABILITY)
        text = pokeapi.ability_info("blaze")
        assert text == pokeapi._format_ability(BLAZE_ABILITY)

    def test_ability_info_not_found(self, monkeypatch):
        monkeypatch.setattr(pokeapi, "_fetch_ability", lambda name: None)
        assert (
            pokeapi.ability_info("not-a-real-ability")
            == "No ability named 'not-a-real-ability' found."
        )

    def test_move_info_found(self, monkeypatch):
        monkeypatch.setattr(pokeapi, "_fetch_move", lambda name: SWORDS_DANCE)
        text = pokeapi.move_info("swords-dance")
        assert text == pokeapi._format_move(SWORDS_DANCE)

    def test_move_info_not_found(self, monkeypatch):
        monkeypatch.setattr(pokeapi, "_fetch_move", lambda name: None)
        assert (
            pokeapi.move_info("not-a-real-move")
            == "No move named 'not-a-real-move' found."
        )

    def test_pokemon_for_type_filters_to_gen_1(self, monkeypatch):
        monkeypatch.setattr(
            pokeapi, "_fetch_type", lambda name: FIRE_TYPE_WITH_POKEMON_AND_MOVES
        )
        names = pokeapi.pokemon_for_type("fire")
        assert set(names) == {"charizard", "charmander"}
        assert "cyndaquil" not in names

    def test_pokemon_for_type_returns_empty_list_for_unknown_type(self, monkeypatch):
        monkeypatch.setattr(pokeapi, "_fetch_type", lambda name: None)
        assert pokeapi.pokemon_for_type("not-a-real-type") == []

    def test_moves_for_type_filters_to_move_names(self, monkeypatch):
        monkeypatch.setattr(
            pokeapi, "_fetch_type", lambda name: FIRE_TYPE_WITH_POKEMON_AND_MOVES
        )
        names = pokeapi.moves_for_type("fire")
        assert set(names) == {"flamethrower", "fire-blast"}
        assert "not-a-real-move" not in names

    def test_moves_for_type_returns_empty_list_for_unknown_type(self, monkeypatch):
        monkeypatch.setattr(pokeapi, "_fetch_type", lambda name: None)
        assert pokeapi.moves_for_type("not-a-real-type") == []

    def test_moves_for_pokemon_includes_red_blue_and_yellow(self, monkeypatch):
        monkeypatch.setattr(
            pokeapi,
            "_fetch_pokemon",
            lambda name: CHARIZARD_POKEMON_WITH_VERSION_DETAILS,
        )
        names = pokeapi.moves_for_pokemon("charizard")
        assert "flamethrower" in names
        assert "growl" in names

    def test_moves_for_pokemon_excludes_other_versions(self, monkeypatch):
        monkeypatch.setattr(
            pokeapi,
            "_fetch_pokemon",
            lambda name: CHARIZARD_POKEMON_WITH_VERSION_DETAILS,
        )
        names = pokeapi.moves_for_pokemon("charizard")
        assert "dragon-claw" not in names

    def test_moves_for_pokemon_dedupes_a_move_listed_under_both_versions(
        self, monkeypatch
    ):
        # "scratch" has both a red-blue and a yellow version_group_details
        # entry -- should appear once, not twice, in the result.
        monkeypatch.setattr(
            pokeapi,
            "_fetch_pokemon",
            lambda name: CHARIZARD_POKEMON_WITH_VERSION_DETAILS,
        )
        names = pokeapi.moves_for_pokemon("charizard")
        assert names.count("scratch") == 1

    def test_moves_for_pokemon_is_sorted(self, monkeypatch):
        monkeypatch.setattr(
            pokeapi,
            "_fetch_pokemon",
            lambda name: CHARIZARD_POKEMON_WITH_VERSION_DETAILS,
        )
        names = pokeapi.moves_for_pokemon("charizard")
        assert names == sorted(names)

    def test_moves_for_pokemon_returns_empty_list_for_unknown_pokemon(
        self, monkeypatch
    ):
        monkeypatch.setattr(pokeapi, "_fetch_pokemon", lambda name: None)
        assert pokeapi.moves_for_pokemon("not-a-real-mon") == []
