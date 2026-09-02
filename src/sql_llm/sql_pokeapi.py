import sqlite3
from pathlib import Path

import pandas as pd

import config

POKEAPI_SCHEMA = """
# PokeAPI SQLite schema — core tables for factual Pokemon questions

## MANDATORY FILTERS — apply to EVERY query unless the question explicitly overrides

- ALWAYS join through pokemon.is_default = 1. Alternate forms (Mega, Primal,
  Gmax, regional) have is_default = 0 and WILL corrupt max/min/count results
  if not excluded. This is the #1 source of wrong answers.
- When the question names a generation ("Generation I", "Gen 3", "Kanto"),
  you MUST filter pokemon_species.generation_id. Gen 1 = 1. Forgetting this
  returns Pokemon from all generations.

## Two rules that govern almost every query

1. NAMES vs IDENTIFIERS. Most tables key on integer IDs. Human-readable
   lowercase slugs live in an `identifier` column (e.g. pokemon_species.identifier
   = 'charizard'). Display names with capitalization/localization live in separate
   `*_names` tables keyed by language (use local_language_id = 9 for English).
   For matching a Pokemon/type/move BY NAME, filter on `identifier` (lowercase),
   e.g. WHERE pokemon_species.identifier = 'charizard'.

2. POKEMON vs SPECIES. `pokemon_species` is the canonical dex entry (Charizard).
   `pokemon` is the concrete form that carries stats/types (Charizard, plus alt
   forms like Mega Charizard X). For normal questions, join species->pokemon on
   pokemon.species_id = pokemon_species.id AND pokemon.is_default = 1 to get the
   base form and avoid double-counting alternate forms.

## Core tables and how they join

pokemon_species(id, identifier, generation_id, evolution_chain_id,
                evolves_from_species_id, color_id, shape_id, habitat_id,
                growth_rate_id, is_legendary, is_mythical, ...)
  - The dex entry. identifier = the name slug ('pikachu').
  - generation_id -> generations.id  (1..9; Gen 1 = 1)
  - evolves_from_species_id -> pokemon_species.id (NULL if base form)
  - color_id -> pokemon_colors.id ; habitat_id -> pokemon_habitats.id ;
    shape_id -> pokemon_shapes.id ; growth_rate_id -> growth_rates.id

pokemon(id, identifier, species_id, height, weight, base_experience, is_default)
  - Concrete form carrying battle data. species_id -> pokemon_species.id
  - height is in decimetres (÷10 for metres); weight in hectograms (÷10 for kg)
  - Filter is_default = 1 for the standard form.

pokemon_stats(pokemon_id, stat_id, base_stat, effort)
  - One row per stat per Pokemon. pokemon_id -> pokemon.id ; stat_id -> stats.id
  - base_stat is the value you aggregate (max/min/avg/sum).
  - Base stat TOTAL = SUM(base_stat) grouped by pokemon_id.

stats(id, identifier, ...)
  - stat_id meanings: 1=hp, 2=attack, 3=defense, 4=special-attack,
    5=special-defense, 6=speed. (identifier holds these slugs.)

pokemon_types(pokemon_id, type_id, slot)
  - A Pokemon has 1-2 types. slot 1 = primary, slot 2 = secondary.
    pokemon_id -> pokemon.id ; type_id -> types.id
types(id, identifier, ...)
  - identifier = 'fire', 'water', ... (18 real types have id 1..18)

type_efficacy(damage_type_id, target_type_id, damage_factor)
  - Attacking-vs-defending multiplier. damage_factor is x100:
    200 = 2x (super effective), 100 = 1x, 50 = 0.5x, 0 = 0x (immune).
  - For a dual-type target, MULTIPLY the two factors (e.g. 200*200/10000 = 4x).
    Both *_type_id columns -> types.id

pokemon_abilities(pokemon_id, ability_id, is_hidden, slot)
  - pokemon_id -> pokemon.id ; ability_id -> abilities.id
  - is_hidden = 1 marks the hidden ability.
abilities(id, identifier, generation_id, ...)  -- identifier = 'blaze', 'static'

pokemon_egg_groups(species_id, egg_group_id)
  - species_id -> pokemon_species.id ; egg_group_id -> egg_groups.id
  - A species can be in 1-2 egg groups.
egg_groups(id, identifier)  -- identifier = 'monster', 'water1', 'ground' (=Field), ...

moves(id, identifier, generation_id, type_id, power, pp, accuracy, priority,
      damage_class_id, ...)
  - type_id -> types.id ; damage_class_id -> move_damage_classes.id
  - move_damage_classes: 1=status, 2=physical, 3=special
  - Z-Moves/Max Moves have pp=1 or identifiers starting 'max-'/'g-max-';
    exclude them for "standard strongest move" questions.

generations(id, identifier, main_region_id)  -- identifier = 'generation-i', ...
  - generation_id lives on pokemon_species, NOT pokemon. To filter by
    generation you MUST join pokemon -> pokemon_species on
    pokemon.species_id = pokemon_species.id, then filter
    pokemon_species.generation_id.

natures(id, identifier, decreased_stat_id, increased_stat_id,
        hates_flavor_id, likes_flavor_id)
  - identifier = 'adamant', 'modest', ... (the nature name slug)
  - increased_stat_id -> stats.id : the stat this nature RAISES
  - decreased_stat_id -> stats.id : the stat this nature LOWERS
  - "raises X and lowers Y": WHERE increased_stat_id = (stat X's id)
    AND decreased_stat_id = (stat Y's id). Join to stats, or use the
    stat_id numbers directly (2=attack, 4=special-attack, etc.).
  - Natures with equal increased/decreased (id equal) are neutral.

experience(growth_rate_id, level, experience)
  - Cumulative EXP to reach a level for a growth rate.
  - "EXP to reach level 100 for <rate>": SELECT experience FROM experience
    JOIN growth_rates ON experience.growth_rate_id = growth_rates.id
    WHERE growth_rates.identifier = '<rate>' AND level = 100.
  - Do NOT compute this from pokemon stats — it's a lookup table.

## base_experience is a COLUMN, not a stat
  - pokemon.base_experience is the EXP yield. It is a plain column on the
    `pokemon` table. Do NOT SUM pokemon_stats, do NOT join to a
    'base_experience' stat row (no such row exists), do NOT join moves.
    Just: SELECT base_experience FROM pokemon ... WHERE identifier = '...'.

## Evolution: use evolves_from_species_id
  - "What does X evolve into?": find species whose evolves_from_species_id
    equals X's species id:
    SELECT identifier FROM pokemon_species
    WHERE evolves_from_species_id = (SELECT id FROM pokemon_species
                                     WHERE identifier = 'x')
  - There is NO evolution_chains/pokemon_evolution table with usable detail
    here — only evolves_from_species_id. Do not reference tables not listed.

## Foreign-key IDs need a JOIN to get the name — never SELECT the raw id
  - color_id, generation_id, egg_group_id, habitat_id etc. are integers.
    "What color is X" must JOIN pokemon_colors and SELECT
    pokemon_colors.identifier — NOT select color_id (which returns '4').
  - Same for generation (join generations -> identifier 'generation-i'),
    egg groups (join egg_groups -> identifier), etc.

## Weight and height conversions — ALWAYS use 10.0, never 10
  - pokemon.weight is in hectograms. To get kilograms, divide by 10.0
    (e.g. weight / 10.0). Use 10.0, NOT 10 — integer division truncates
    (855/10 = 85, but 855/10.0 = 85.5).
  - pokemon.height is in decimetres. For metres, divide by 10.0 likewise.

## Exact egg_group identifiers (they are NOT the display names — use these):
  monster, water1, water2, water3, bug, flying, ground (=Field),
  fairy, plant (=Grass), humanshape (=Human-Like), mineral,
  indeterminate (=Amorphous), ditto, dragon, no-eggs (=Undiscovered).
  Match on these exact strings, e.g. WHERE egg_groups.identifier='humanshape'
  (NOT 'human-shape' or 'human-shaped').

## The ONLY tables that exist (do not reference any others):
  pokemon, pokemon_species, pokemon_stats, stats, types, pokemon_types,
  type_efficacy, abilities, pokemon_abilities, egg_groups, pokemon_egg_groups,
  moves, move_damage_classes, generations, growth_rates, experience, natures,
  pokemon_colors, pokemon_shapes, pokemon_habitats, berries, berry_firmness,
  berry_flavors, items, item_categories, item_pockets, regions.

## Common patterns

- Name a Pokemon in results: join to pokemon_species and select identifier.
- "Gen N X-type": join pokemon_species (generation_id=N) -> pokemon (is_default=1)
  -> pokemon_types -> types (identifier='x').
- "Highest base <stat>": join to pokemon_stats where stats.identifier='<stat>',
  ORDER BY base_stat DESC LIMIT 1.
- "Base stat total": SUM(pokemon_stats.base_stat) GROUP BY pokemon_id.
- Always join through pokemon.is_default = 1 unless the question is about alt forms.
"""


def build_database(csv_dir: str, db_path: str = "pokeapi.db"):
    """Load every CSV in csv_dir into a SQLite database, one table per file.
    Table names are the CSV filenames without extension (e.g. pokemon.csv -> pokemon)."""
    csv_dir = Path(csv_dir)
    csv_files = sorted(csv_dir.glob("*.csv"))
    if not csv_files:
        raise FileNotFoundError(f"No CSVs found in {csv_dir}")

    # fresh build each time so re-running doesn't stack duplicate rows
    conn = sqlite3.connect(db_path)
    loaded, skipped = [], []

    for csv_path in csv_files:
        table = csv_path.stem
        try:
            df = pd.read_csv(csv_path)
            df.to_sql(table, conn, if_exists="replace", index=False)
            loaded.append((table, len(df)))
        except Exception as e:
            skipped.append((table, str(e)))

    conn.commit()
    conn.close()

    print(f"Loaded {len(loaded)} tables into {db_path}:")
    for name, n in sorted(loaded):
        print(f"  {name:<40} {n:>7} rows")
    if skipped:
        print(f"\nSkipped {len(skipped)}:")
        for name, err in skipped:
            print(f"  {name}: {err[:80]}")

    return db_path


def get_readonly_connection(db_path: str = "pokeapi.db"):
    """Open a READ-ONLY connection — model-generated SQL cannot modify the DB."""
    # file: URI with mode=ro enforces read-only at the SQLite level
    uri = f"file:{Path(db_path).resolve()}?mode=ro"
    return sqlite3.connect(uri, uri=True)


def run_query(sql: str, db_path: str = "pokeapi.db", timeout_s: float = 5.0):
    """Execute a read-only query with a timeout. Returns (columns, rows) or raises."""
    conn = get_readonly_connection(db_path)
    try:
        conn.execute(f"PRAGMA busy_timeout = {int(timeout_s * 1000)}")
        cur = conn.execute(sql)
        columns = [d[0] for d in cur.description] if cur.description else []
        rows = cur.fetchall()
        return columns, rows
    finally:
        conn.close()


def get_schema(db_path: str = "pokeapi.db") -> str:
    """Return a text description of all tables and columns — feed this to the
    text-to-SQL model so it knows what it can query."""
    conn = get_readonly_connection(db_path)
    try:
        tables = [
            r[0]
            for r in conn.execute(
                "SELECT name FROM sqlite_master WHERE type='table' ORDER BY name"
            )
        ]
        lines = []
        for t in tables:
            cols = conn.execute(f"PRAGMA table_info({t})").fetchall()
            col_str = ", ".join(f"{c[1]}" for c in cols)
            lines.append(f"{t}({col_str})")
        return "\n".join(lines)
    finally:
        conn.close()


if __name__ == "__main__":
    build_database(config.ROOT_DIR / "data" / "pokeapi_data")
    print("\n--- schema preview ---")
    print(get_schema()[:2000])
