"""Live PokeAPI-backed retrieval — replaces src/vectorstore.py's embedding +
Chroma similarity search with a direct name-match + REST lookup against the
live PokeAPI. No local index, no embedding model, no bulk data download.

src/vectorstore.py and src/ingest.py are left in place, unused by the
current pipeline, for whenever the full bulk dataset gets adopted later.

Covers roadmap steps 1-6 from broad_question.md: pokemon-species + pokemon
(stats/types/abilities/moves), evolution chains, types, abilities, moves,
and items. Step 7 (locations) and step 8 (relational questions) aren't here.
"""

import json
import urllib.error
import urllib.request

import config

# National Dex #1-151, PokeAPI name slugs (lowercase, hyphenated e.g.
# "mr-mime", "nidoran-f"). Fill in before pokeapi.query() will match anything.
GEN_1_NAMES = [
    "bulbasaur",
    "ivysaur",
    "venusaur",
    "charmander",
    "charmeleon",
    "charizard",
    "squirtle",
    "wartortle",
    "blastoise",
    "caterpie",
    "metapod",
    "butterfree",
    "weedle",
    "kakuna",
    "beedrill",
    "pidgey",
    "pidgeotto",
    "pidgeot",
    "rattata",
    "raticate",
    "spearow",
    "fearow",
    "ekans",
    "arbok",
    "pikachu",
    "raichu",
    "sandshrew",
    "sandslash",
    "nidoran-f",
    "nidorina",
    "nidoqueen",
    "nidoran-m",
    "nidorino",
    "nidoking",
    "clefairy",
    "clefable",
    "vulpix",
    "ninetales",
    "jigglypuff",
    "wigglytuff",
    "zubat",
    "golbat",
    "oddish",
    "gloom",
    "vileplume",
    "paras",
    "parasect",
    "venonat",
    "venomoth",
    "diglett",
    "dugtrio",
    "meowth",
    "persian",
    "psyduck",
    "golduck",
    "mankey",
    "primeape",
    "growlithe",
    "arcanine",
    "poliwag",
    "poliwhirl",
    "poliwrath",
    "abra",
    "kadabra",
    "alakazam",
    "machop",
    "machoke",
    "machamp",
    "bellsprout",
    "weepinbell",
    "victreebel",
    "tentacool",
    "tentacruel",
    "geodude",
    "graveler",
    "golem",
    "ponyta",
    "rapidash",
    "slowpoke",
    "slowbro",
    "magnemite",
    "magneton",
    "farfetchd",
    "doduo",
    "dodrio",
    "seel",
    "dewgong",
    "grimer",
    "muk",
    "shellder",
    "cloyster",
    "gastly",
    "haunter",
    "gengar",
    "onix",
    "drowzee",
    "hypno",
    "krabby",
    "kingler",
    "voltorb",
    "electrode",
    "exeggcute",
    "exeggutor",
    "cubone",
    "marowak",
    "hitmonlee",
    "hitmonchan",
    "lickitung",
    "koffing",
    "weezing",
    "rhyhorn",
    "rhydon",
    "chansey",
    "tangela",
    "kangaskhan",
    "horsea",
    "seadra",
    "goldeen",
    "seaking",
    "staryu",
    "starmie",
    "mr-mime",
    "scyther",
    "jynx",
    "electabuzz",
    "magmar",
    "pinsir",
    "tauros",
    "magikarp",
    "gyarados",
    "lapras",
    "ditto",
    "eevee",
    "vaporeon",
    "jolteon",
    "flareon",
    "porygon",
    "omanyte",
    "omastar",
    "kabuto",
    "kabutops",
    "aerodactyl",
    "snorlax",
    "articuno",
    "zapdos",
    "moltres",
    "dratini",
    "dragonair",
    "dragonite",
    "mewtwo",
    "mew",
]

# The 18 PokeAPI type slugs. Fixed and exhaustive (unlike the lists below),
# so this one's safe to hardcode in full.
TYPE_NAMES = [
    "normal",
    "fire",
    "water",
    "electric",
    "grass",
    "ice",
    "fighting",
    "poison",
    "ground",
    "flying",
    "psychic",
    "bug",
    "rock",
    "ghost",
    "dragon",
    "dark",
    "steel",
    "fairy",
]

# Abilities worth answering standalone questions about ("what does Blaze
# do"). Left empty like GEN_1_NAMES — fill in once GEN_1_NAMES is populated
# and you can see which abilities your gen-1 set actually has.
ABILITY_NAMES = [
    "adaptability",
    "aftermath",
    "analytic",
    "anger-point",
    "anticipation",
    "arena-trap",
    "battle-armor",
    "big-pecks",
    "blaze",
    "chlorophyll",
    "clear-body",
    "cloud-nine",
    "competitive",
    "compound-eyes",
    "cursed-body",
    "cute-charm",
    "damp",
    "defiant",
    "download",
    "drought",
    "dry-skin",
    "early-bird",
    "effect-spore",
    "filter",
    "flame-body",
    "flash-fire",
    "forewarn",
    "friend-guard",
    "frisk",
    "gluttony",
    "guts",
    "harvest",
    "healer",
    "hustle",
    "hydration",
    "hyper-cutter",
    "ice-body",
    "illuminate",
    "immunity",
    "imposter",
    "infiltrator",
    "inner-focus",
    "insomnia",
    "intimidate",
    "iron-fist",
    "justified",
    "keen-eye",
    "leaf-guard",
    "levitate",
    "lightning-rod",
    "limber",
    "liquid-ooze",
    "magic-guard",
    "magnet-pull",
    "marvel-scale",
    "mold-breaker",
    "moxie",
    "multiscale",
    "natural-cure",
    "neutralizing-gas",
    "no-guard",
    "oblivious",
    "overcoat",
    "overgrow",
    "own-tempo",
    "pickup",
    "poison-point",
    "poison-touch",
    "pressure",
    "quick-feet",
    "rain-dish",
    "rattled",
    "reckless",
    "regenerator",
    "rivalry",
    "rock-head",
    "run-away",
    "sand-force",
    "sand-rush",
    "sand-veil",
    "scrappy",
    "serene-grace",
    "shed-skin",
    "sheer-force",
    "shell-armor",
    "shield-dust",
    "skill-link",
    "sniper",
    "snow-cloak",
    "solar-power",
    "soundproof",
    "static",
    "steadfast",
    "stench",
    "sticky-hold",
    "sturdy",
    "swarm",
    "swift-swim",
    "synchronize",
    "tangled-feet",
    "technician",
    "thick-fat",
    "tinted-lens",
    "torrent",
    "trace",
    "unaware",
    "unburden",
    "unnerve",
    "vital-spirit",
    "volt-absorb",
    "water-absorb",
    "water-veil",
    "weak-armor",
    "wonder-skin",
]

# Same idea as ABILITY_NAMES, scoped to moves your gen-1 set can learn.
MOVE_NAMES = [
    "absorb",
    "acid",
    "acid-armor",
    "acid-spray",
    "acrobatics",
    "acupressure",
    "aerial-ace",
    "after-you",
    "agility",
    "air-cutter",
    "air-slash",
    "alluring-voice",
    "ally-switch",
    "amnesia",
    "ancient-power",
    "aqua-jet",
    "aqua-ring",
    "aqua-tail",
    "aromatherapy",
    "assist",
    "assurance",
    "astonish",
    "attract",
    "aura-sphere",
    "aurora-beam",
    "aurora-veil",
    "autotomize",
    "avalanche",
    "axe-kick",
    "baby-doll-eyes",
    "barrage",
    "barrier",
    "baton-pass",
    "beat-up",
    "belch",
    "belly-drum",
    "bestow",
    "bide",
    "bind",
    "bite",
    "blast-burn",
    "blaze-kick",
    "blizzard",
    "block",
    "body-press",
    "body-slam",
    "bone-club",
    "bone-rush",
    "bonemerang",
    "bounce",
    "brave-bird",
    "breaking-swipe",
    "brick-break",
    "brine",
    "brutal-swing",
    "bubble",
    "bubble-beam",
    "bug-bite",
    "bug-buzz",
    "bulk-up",
    "bulldoze",
    "bullet-punch",
    "bullet-seed",
    "burn-up",
    "burning-jealousy",
    "calm-mind",
    "camouflage",
    "captivate",
    "charge",
    "charge-beam",
    "charm",
    "chilling-water",
    "chip-away",
    "circle-throw",
    "clamp",
    "clear-smog",
    "close-combat",
    "coaching",
    "coil",
    "comet-punch",
    "confide",
    "confuse-ray",
    "confusion",
    "constrict",
    "conversion",
    "conversion-2",
    "copycat",
    "corrosive-gas",
    "cosmic-power",
    "counter",
    "covet",
    "crabhammer",
    "cross-chop",
    "cross-poison",
    "crunch",
    "crush-claw",
    "curse",
    "cut",
    "dark-pulse",
    "darkest-lariat",
    "dazzling-gleam",
    "defense-curl",
    "defog",
    "destiny-bond",
    "detect",
    "dig",
    "disable",
    "disarming-voice",
    "discharge",
    "dive",
    "dizzy-punch",
    "double-edge",
    "double-hit",
    "double-kick",
    "double-slap",
    "double-team",
    "draco-meteor",
    "dragon-breath",
    "dragon-cheer",
    "dragon-claw",
    "dragon-dance",
    "dragon-pulse",
    "dragon-rage",
    "dragon-rush",
    "dragon-tail",
    "drain-punch",
    "draining-kiss",
    "dream-eater",
    "drill-peck",
    "drill-run",
    "dual-chop",
    "dual-wingbeat",
    "dynamic-punch",
    "earth-power",
    "earthquake",
    "echoed-voice",
    "eerie-impulse",
    "egg-bomb",
    "electric-terrain",
    "electro-ball",
    "electroweb",
    "embargo",
    "ember",
    "encore",
    "endeavor",
    "endure",
    "energy-ball",
    "entrainment",
    "expanding-force",
    "explosion",
    "extrasensory",
    "extreme-speed",
    "facade",
    "fairy-wind",
    "fake-out",
    "fake-tears",
    "false-swipe",
    "feather-dance",
    "feint",
    "feint-attack",
    "fell-stinger",
    "final-gambit",
    "fire-blast",
    "fire-fang",
    "fire-pledge",
    "fire-punch",
    "fire-spin",
    "first-impression",
    "fissure",
    "flail",
    "flame-burst",
    "flame-charge",
    "flame-wheel",
    "flamethrower",
    "flare-blitz",
    "flash",
    "flash-cannon",
    "flatter",
    "fling",
    "flip-turn",
    "fly",
    "focus-blast",
    "focus-energy",
    "focus-punch",
    "follow-me",
    "foresight",
    "foul-play",
    "freeze-dry",
    "frenzy-plant",
    "frost-breath",
    "frustration",
    "fury-attack",
    "fury-cutter",
    "fury-swipes",
    "future-sight",
    "gastro-acid",
    "giga-drain",
    "giga-impact",
    "glare",
    "grass-knot",
    "grass-pledge",
    "grass-whistle",
    "grassy-glide",
    "grassy-terrain",
    "gravity",
    "growl",
    "growth",
    "grudge",
    "guard-split",
    "guard-swap",
    "guillotine",
    "gunk-shot",
    "gust",
    "gyro-ball",
    "hail",
    "hammer-arm",
    "hard-press",
    "harden",
    "haze",
    "head-smash",
    "headbutt",
    "heal-bell",
    "heal-pulse",
    "healing-wish",
    "heart-stamp",
    "heat-crash",
    "heat-wave",
    "heavy-slam",
    "helping-hand",
    "hex",
    "hidden-power",
    "high-horsepower",
    "high-jump-kick",
    "hone-claws",
    "horn-attack",
    "horn-drill",
    "howl",
    "hurricane",
    "hydro-cannon",
    "hydro-pump",
    "hyper-beam",
    "hyper-fang",
    "hyper-voice",
    "hypnosis",
    "ice-ball",
    "ice-beam",
    "ice-fang",
    "ice-punch",
    "ice-shard",
    "ice-spinner",
    "icicle-crash",
    "icicle-spear",
    "icy-wind",
    "imprison",
    "incinerate",
    "inferno",
    "infestation",
    "ingrain",
    "iron-defense",
    "iron-head",
    "iron-tail",
    "jump-kick",
    "karate-chop",
    "kinesis",
    "knock-off",
    "laser-focus",
    "lash-out",
    "last-resort",
    "lava-plume",
    "leaf-blade",
    "leaf-storm",
    "leaf-tornado",
    "leech-life",
    "leech-seed",
    "leer",
    "lick",
    "life-dew",
    "light-screen",
    "liquidation",
    "lock-on",
    "lovely-kiss",
    "low-kick",
    "low-sweep",
    "lucky-chant",
    "lunge",
    "mach-punch",
    "magic-coat",
    "magic-room",
    "magical-leaf",
    "magnet-bomb",
    "magnet-rise",
    "magnetic-flux",
    "magnitude",
    "me-first",
    "mean-look",
    "meditate",
    "mega-drain",
    "mega-kick",
    "mega-punch",
    "megahorn",
    "memento",
    "metal-burst",
    "metal-claw",
    "metal-sound",
    "meteor-beam",
    "meteor-mash",
    "metronome",
    "mimic",
    "mind-reader",
    "minimize",
    "miracle-eye",
    "mirror-coat",
    "mirror-move",
    "mirror-shot",
    "mist",
    "misty-explosion",
    "misty-terrain",
    "moonblast",
    "moonlight",
    "morning-sun",
    "mud-bomb",
    "mud-shot",
    "mud-slap",
    "mud-sport",
    "muddy-water",
    "mystical-fire",
    "nasty-plot",
    "natural-gift",
    "nature-power",
    "night-shade",
    "night-slash",
    "nightmare",
    "nuzzle",
    "octazooka",
    "odor-sleuth",
    "ominous-wind",
    "outrage",
    "overheat",
    "pain-split",
    "pay-day",
    "payback",
    "peck",
    "perish-song",
    "petal-blizzard",
    "petal-dance",
    "phantom-force",
    "pin-missile",
    "play-nice",
    "play-rough",
    "pluck",
    "poison-fang",
    "poison-gas",
    "poison-jab",
    "poison-powder",
    "poison-sting",
    "poison-tail",
    "pollen-puff",
    "poltergeist",
    "pounce",
    "pound",
    "powder-snow",
    "power-gem",
    "power-split",
    "power-swap",
    "power-trick",
    "power-trip",
    "power-up-punch",
    "power-whip",
    "present",
    "protect",
    "psybeam",
    "psych-up",
    "psychic",
    "psychic-fangs",
    "psychic-noise",
    "psychic-terrain",
    "psycho-cut",
    "psycho-shift",
    "psyshock",
    "psystrike",
    "psywave",
    "punishment",
    "pursuit",
    "quash",
    "quick-attack",
    "quick-guard",
    "quiver-dance",
    "rage",
    "rage-fist",
    "rage-powder",
    "raging-bull",
    "raging-fury",
    "rain-dance",
    "rapid-spin",
    "razor-leaf",
    "razor-shell",
    "razor-wind",
    "recover",
    "recycle",
    "reflect",
    "reflect-type",
    "refresh",
    "rest",
    "retaliate",
    "return",
    "revenge",
    "reversal",
    "rising-voltage",
    "roar",
    "rock-blast",
    "rock-climb",
    "rock-polish",
    "rock-slide",
    "rock-smash",
    "rock-throw",
    "rock-tomb",
    "role-play",
    "rolling-kick",
    "rollout",
    "roost",
    "rototiller",
    "round",
    "safeguard",
    "sand-attack",
    "sand-tomb",
    "sandstorm",
    "scald",
    "scale-shot",
    "scary-face",
    "scorching-sands",
    "scratch",
    "screech",
    "secret-power",
    "seed-bomb",
    "seismic-toss",
    "self-destruct",
    "shadow-ball",
    "shadow-claw",
    "shadow-punch",
    "shadow-sneak",
    "sharpen",
    "sheer-cold",
    "shell-smash",
    "shock-wave",
    "signal-beam",
    "silver-wind",
    "simple-beam",
    "sing",
    "skill-swap",
    "skitter-smack",
    "skull-bash",
    "sky-attack",
    "sky-drop",
    "sky-uppercut",
    "slack-off",
    "slam",
    "slash",
    "sleep-powder",
    "sleep-talk",
    "sludge",
    "sludge-bomb",
    "sludge-wave",
    "smack-down",
    "smart-strike",
    "smelling-salts",
    "smog",
    "smokescreen",
    "snarl",
    "snatch",
    "snore",
    "snowscape",
    "soak",
    "soft-boiled",
    "solar-beam",
    "solar-blade",
    "sonic-boom",
    "spark",
    "sparkling-aria",
    "speed-swap",
    "spike-cannon",
    "spikes",
    "spit-up",
    "spite",
    "splash",
    "spore",
    "spotlight",
    "stealth-rock",
    "steamroller",
    "steel-beam",
    "steel-roller",
    "steel-wing",
    "stockpile",
    "stomp",
    "stomping-tantrum",
    "stone-edge",
    "stored-power",
    "storm-throw",
    "strength",
    "strength-sap",
    "string-shot",
    "struggle-bug",
    "stun-spore",
    "submission",
    "substitute",
    "sucker-punch",
    "sunny-day",
    "super-fang",
    "supercell-slam",
    "superpower",
    "supersonic",
    "surf",
    "swagger",
    "swallow",
    "sweet-kiss",
    "sweet-scent",
    "swift",
    "switcheroo",
    "swords-dance",
    "synchronoise",
    "synthesis",
    "tackle",
    "tail-slap",
    "tail-whip",
    "tailwind",
    "take-down",
    "taunt",
    "teeter-dance",
    "telekinesis",
    "teleport",
    "temper-flare",
    "tera-blast",
    "terrain-pulse",
    "thief",
    "thrash",
    "throat-chop",
    "thunder",
    "thunder-fang",
    "thunder-punch",
    "thunder-shock",
    "thunder-wave",
    "thunderbolt",
    "tickle",
    "torment",
    "toxic",
    "toxic-spikes",
    "trailblaze",
    "transform",
    "tri-attack",
    "trick",
    "trick-room",
    "triple-axel",
    "trump-card",
    "twineedle",
    "twister",
    "u-turn",
    "upper-hand",
    "uproar",
    "vacuum-wave",
    "venom-drench",
    "venoshock",
    "vice-grip",
    "vine-whip",
    "vital-throw",
    "volt-switch",
    "volt-tackle",
    "wake-up-slap",
    "water-gun",
    "water-pledge",
    "water-pulse",
    "water-sport",
    "water-spout",
    "waterfall",
    "wave-crash",
    "weather-ball",
    "whirlpool",
    "whirlwind",
    "wide-guard",
    "wild-charge",
    "will-o-wisp",
    "wing-attack",
    "wish",
    "withdraw",
    "wonder-room",
    "wood-hammer",
    "work-up",
    "worry-seed",
    "wrap",
    "wring-out",
    "x-scissor",
    "yawn",
    "zap-cannon",
    "zen-headbutt",
]

ITEM_NAMES = [
    "absorb-bulb",
    "aspear-berry",
    "balm-mushroom",
    "big-mushroom",
    "big-pearl",
    "black-belt",
    "black-sludge",
    "charcoal",
    "chesto-berry",
    "chilan-berry",
    "comet-shard",
    "dragon-fang",
    "dragon-scale",
    "electirizer",
    "everstone",
    "focus-band",
    "grip-claw",
    "hard-stone",
    "kings-rock",
    "lagging-tail",
    "leftovers",
    "leppa-berry",
    "light-ball",
    "lucky-egg",
    "lucky-punch",
    "lum-berry",
    "magmarizer",
    "magnet",
    "metal-coat",
    "metal-powder",
    "miracle-seed",
    "moon-stone",
    "mystic-water",
    "never-melt-ice",
    "nugget",
    "oran-berry",
    "oval-stone",
    "payapa-berry",
    "pearl",
    "poison-barb",
    "psychic-seed",
    "quick-claw",
    "quick-powder",
    "rawst-berry",
    "sharp-beak",
    "shed-shell",
    "shuca-berry",
    "silver-powder",
    "sitrus-berry",
    "smoke-ball",
    "soft-sand",
    "spell-tag",
    "star-piece",
    "stardust",
    "stick",
    "thick-club",
    "tiny-mushroom",
    "toxic-orb",
    "twisted-spoon",
]

_cache = {}


def _fetch(url):
    if url in _cache:
        return _cache[url]

    req = urllib.request.Request(url, headers={"User-Agent": "my-project/1.0"})
    try:
        with urllib.request.urlopen(req, timeout=10) as resp:
            data = json.loads(resp.read())
    except urllib.error.HTTPError as e:
        if e.code == 404:
            return None
        raise

    _cache[url] = data
    return data


def _fetch_resource(resource, name):
    return _fetch(f"{config.POKEAPI_BASE_URL}/{resource}/{name}")


def _english(entries, field):
    return next((e[field] for e in entries if e["language"]["name"] == "en"), "")


def _match(names, lowered):
    return [
        name for name in names if name in lowered or name.replace("-", " ") in lowered
    ]


# --- pokemon-species: genus + flavor text ----------------------------------


def _fetch_species(name):
    return _fetch_resource("pokemon-species", name)


def _format_species(data):
    genus = _english(data["genera"], "genus")

    seen = set()
    flavor_lines = []
    for entry in data["flavor_text_entries"]:
        if entry["language"]["name"] != "en":
            continue
        text = " ".join(entry["flavor_text"].split())
        if text not in seen:
            seen.add(text)
            flavor_lines.append(text)

    parts = ([f"{genus}."] if genus else []) + flavor_lines
    return " ".join(parts).strip()


# --- pokemon: types, abilities, stats, moves --------------------------------


def _fetch_pokemon(name):
    return _fetch_resource("pokemon", name)


def _format_pokemon(data):
    types = ", ".join(
        t["type"]["name"] for t in sorted(data["types"], key=lambda t: t["slot"])
    )
    abilities = ", ".join(
        a["ability"]["name"] + (" (hidden)" if a["is_hidden"] else "")
        for a in data["abilities"]
    )
    stats = ", ".join(f"{s['stat']['name']} {s['base_stat']}" for s in data["stats"])
    moves = sorted({m["move"]["name"] for m in data["moves"]})

    return (
        f"Types: {types}. Abilities: {abilities}. Base stats: {stats}. "
        f"Moves it can learn: {', '.join(moves)}."
    )


# --- evolution chain --------------------------------------------------------


def _fetch_evolution_chain(url):
    return _fetch(url)


def _describe_evolution_trigger(details):
    if not details:
        return "somehow"
    d = details[0]
    trigger = d["trigger"]["name"] if d["trigger"] else "unknown"
    if trigger == "level-up" and d.get("min_level"):
        return f"at level {d['min_level']}"
    if trigger == "use-item" and d.get("item"):
        return f"using a {d['item']['name']}"
    if trigger == "trade":
        return "by trading"
    return trigger.replace("-", " ")


def _format_evolution_chain(data):
    lines = []

    def walk(node):
        from_name = node["species"]["name"]
        for evo in node["evolves_to"]:
            to_name = evo["species"]["name"]
            condition = _describe_evolution_trigger(evo["evolution_details"])
            lines.append(f"{from_name} evolves into {to_name} {condition}.")
            walk(evo)

    walk(data["chain"])
    return " ".join(lines) if lines else "This Pokemon does not evolve."


# --- type: damage relations -------------------------------------------------


def _fetch_type(name):
    return _fetch_resource("type", name)


def _format_type(data):
    rel = data["damage_relations"]

    def names(key):
        return ", ".join(t["name"] for t in rel[key]) or "none"

    return (
        f"{data['name'].capitalize()} type deals double damage to: "
        f"{names('double_damage_to')}. Deals half damage to: "
        f"{names('half_damage_to')}. Deals no damage to: "
        f"{names('no_damage_to')}. Takes double damage from: "
        f"{names('double_damage_from')}. Takes half damage from: "
        f"{names('half_damage_from')}. Takes no damage from: "
        f"{names('no_damage_from')}."
    )


# --- ability -----------------------------------------------------------------


def _fetch_ability(name):
    return _fetch_resource("ability", name)


def _format_ability(data):
    return " ".join(_english(data["effect_entries"], "effect").split())


# --- move ----------------------------------------------------------------------


def _fetch_move(name):
    return _fetch_resource("move", name)


def _format_move(data):
    effect = " ".join(_english(data["effect_entries"], "effect").split())
    return (
        f"{data['name'].replace('-', ' ').capitalize()} is a {data['type']['name']}-type "
        f"{data['damage_class']['name']} move. Power {data['power']}, accuracy "
        f"{data['accuracy']}, PP {data['pp']}. {effect}"
    )


# --- item ----------------------------------------------------------------------


def _fetch_item(name):
    return _fetch_resource("item", name)


def _format_item(data):
    return " ".join(_english(data["effect_entries"], "effect").split())


# --- ReAct tool functions (broad_question.md Approach A) --------------------
# One callable per resource, wrapping the existing _fetch_* + _format_* pair.
# Docstrings describe the argument and return value (not the implementation)
# since these are what dspy.ReAct reads to build its tool schema.


def pokemon_for_type(type_name: str) -> list[str]:
    """Look up which Gen 1 Pokemon have a given type.

    Args:
        type_name: A PokeAPI type name (e.g. "fire", "flying").

    Returns:
        A list of Gen 1 Pokemon name slugs with that type, or an empty list
        if no type was found under that name.
    """
    type_data = _fetch_type(type_name)
    if type_data is None:
        return []
    pokemon_with_type = type_data["pokemon"]
    return [
        pkmn_data["pokemon"]["name"]
        for pkmn_data in pokemon_with_type
        if pkmn_data["pokemon"]["name"] in GEN_1_NAMES
    ]


def moves_for_type(type_name: str) -> list[str]:
    """Look up which moves in this project's move set have a given type.

    Args:
        type_name: A PokeAPI type name (e.g. "fire", "flying").

    Returns:
        A list of move name slugs of that type, or an empty list if no type
        was found under that name.
    """
    type_data = _fetch_type(type_name)
    if type_data is None:
        return []
    moves_with_type = type_data["moves"]
    return [
        move_data["name"]
        for move_data in moves_with_type
        if move_data["name"] in MOVE_NAMES
    ]


def moves_for_pokemon(name: str) -> list[str]:
    """Look up which moves a Pokemon can learn in Red, Blue, or Yellow.

    Args:
        name: A PokeAPI Pokemon name slug (lowercase, hyphenated, e.g.
            "charizard", "mr-mime").

    Returns:
        A sorted list of move name slugs learnable in the Red, Blue, or
        Yellow version group, or an empty list if no Pokemon was found
        under that name.
    """
    pkmn_data = _fetch_pokemon(name)
    if pkmn_data is None:
        return []
    moves_data = pkmn_data["moves"]
    moves = set()
    for move in moves_data:
        version_details = move["version_group_details"]
        for detail in version_details:
            if (
                "red-blue" in detail["version_group"]["name"]
                or "yellow" in detail["version_group"]["name"]
            ):
                moves.add(move["move"]["name"])
    return sorted(moves)


def pokemon_info(name: str) -> str:
    """Look up a Pokemon's types, abilities, base stats, and learnable moves.

    Args:
        name: A PokeAPI Pokemon name slug (lowercase, hyphenated, e.g.
            "charizard", "mr-mime").

    Returns:
        A string describing the Pokemon's types, abilities, base stats, and
        moves, or a message saying no Pokemon was found under that name.
    """
    data = _fetch_pokemon(name)
    if data is None:
        return f"No Pokemon named '{name}' found."
    return _format_pokemon(data)


def species_info(name: str) -> str:
    """Look up a Pokemon species' genus and Pokedex flavor text.

    Args:
        name: A PokeAPI Pokemon name slug (lowercase, hyphenated, e.g.
            "charizard", "mr-mime").

    Returns:
        A string with the Pokemon's genus and flavor text, or a message
        saying no species was found under that name.
    """
    data = _fetch_species(name)
    if data is None:
        return f"No Pokemon species named '{name}' found."
    return _format_species(data)


def evolution_chain_for(name: str) -> str:
    """Look up what a Pokemon evolves from and into, and the conditions.

    Args:
        name: A PokeAPI Pokemon name slug (lowercase, hyphenated, e.g.
            "charmander").

    Returns:
        A string describing every evolution step in this Pokemon's family
        line, or a message saying no species/evolution chain was found.
    """
    species = _fetch_species(name)
    if species is None:
        return f"No Pokemon species named '{name}' found."

    evolution_chain = species.get("evolution_chain")
    if not evolution_chain:
        return f"'{name}' has no evolution chain data."

    chain = _fetch_evolution_chain(evolution_chain["url"])
    if chain is None:
        return f"Could not fetch the evolution chain for '{name}'."
    return _format_evolution_chain(chain)


def type_relations(name: str) -> str:
    """Look up a type's damage relations (what it's strong/weak against).

    Args:
        name: A PokeAPI type name (e.g. "fire", "flying").

    Returns:
        A string describing double/half/no damage dealt and taken, or a
        message saying no type was found under that name.
    """
    data = _fetch_type(name)
    if data is None:
        return f"No type named '{name}' found."
    return _format_type(data)


def ability_info(name: str) -> str:
    """Look up what a Pokemon ability does.

    Args:
        name: A PokeAPI ability name slug (lowercase, hyphenated, e.g.
            "blaze", "solar-power").

    Returns:
        A string describing the ability's effect, or a message saying no
        ability was found under that name.
    """
    data = _fetch_ability(name)
    if data is None:
        return f"No ability named '{name}' found."
    return _format_ability(data)


def move_info(name: str) -> str:
    """Look up a move's type, power, accuracy, PP, and effect.

    Args:
        name: A PokeAPI move name slug (lowercase, hyphenated, e.g.
            "flamethrower", "swords-dance").

    Returns:
        A string describing the move, or a message saying no move was found
        under that name.
    """
    data = _fetch_move(name)
    if data is None:
        return f"No move named '{name}' found."
    return _format_move(data)


# --- orchestration -----------------------------------------------------------


def query(text, top_k=config.TOP_K):
    """Match known names in `text` against every resource list, fetch +
    format live PokeAPI data for each match, and merge into one hit list.

    Returns the same [{"text", "source", "score"}, ...] shape as
    src/vectorstore.query, so src/rag.py and src/dspy_eval.py don't need to
    change as more resource types get added here. `top_k` caps how many
    matches get looked up per resource list (species/pokemon/evolution
    count as one GEN_1_NAMES match), not a single global result count.
    """
    lowered = text.lower()
    hits = []

    for name in _match(GEN_1_NAMES, lowered)[:top_k]:
        species = _fetch_species(name)
        if species is not None:
            hits.append(
                {
                    "text": _format_species(species),
                    "source": f"pokeapi:species:{name}",
                    "score": 1.0,
                }
            )

        pokemon = _fetch_pokemon(name)
        if pokemon is not None:
            hits.append(
                {
                    "text": _format_pokemon(pokemon),
                    "source": f"pokeapi:pokemon:{name}",
                    "score": 1.0,
                }
            )

        if (
            species is not None
            and " evol" in lowered
            and species.get("evolution_chain")
        ):
            chain = _fetch_evolution_chain(species["evolution_chain"]["url"])
            if chain is not None:
                hits.append(
                    {
                        "text": _format_evolution_chain(chain),
                        "source": f"pokeapi:evolution-chain:{name}",
                        "score": 1.0,
                    }
                )

    for name in _match(TYPE_NAMES, lowered)[:top_k]:
        data = _fetch_type(name)
        if data is not None:
            hits.append(
                {
                    "text": _format_type(data),
                    "source": f"pokeapi:type:{name}",
                    "score": 1.0,
                }
            )

    for name in _match(ABILITY_NAMES, lowered)[:top_k]:
        data = _fetch_ability(name)
        if data is not None:
            hits.append(
                {
                    "text": _format_ability(data),
                    "source": f"pokeapi:ability:{name}",
                    "score": 1.0,
                }
            )

    for name in _match(MOVE_NAMES, lowered)[:top_k]:
        data = _fetch_move(name)
        if data is not None:
            hits.append(
                {
                    "text": _format_move(data),
                    "source": f"pokeapi:move:{name}",
                    "score": 1.0,
                }
            )

    for name in _match(ITEM_NAMES, lowered)[:top_k]:
        data = _fetch_item(name)
        if data is not None:
            hits.append(
                {
                    "text": _format_item(data),
                    "source": f"pokeapi:item:{name}",
                    "score": 1.0,
                }
            )

    return hits


if __name__ == "__main__":
    import sys

    q = " ".join(sys.argv[1:]) or "What type is Charizard and what does it evolve from?"
    print(query(q))
