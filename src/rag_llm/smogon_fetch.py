import html
import json
import re
import time
import urllib.request
from pathlib import Path

from langchain_text_splitters import RecursiveCharacterTextSplitter

import config

SETS_STORE = config.ROOT_DIR / "data/raw/smogon/sets.json"
ANALYSES_DIR = config.DATA_RAW_DIR / "smogon" / "analyses"


def _clean_html(text):
    text = re.sub(r"<[^>]+>", " ", text)  # strip tags
    text = html.unescape(text)  # decode entities
    return re.sub(r"\s+", " ", text).strip()  # collapse whitespace


def _opt(v, joiner=" / "):
    """Render a field that may be a single value or a list of alternatives."""
    if v is None:
        return "any"
    if isinstance(v, list):
        # list of dicts (evs) vs list of strings (moves/items/natures)
        if v and isinstance(v[0], dict):
            return " OR ".join(
                ", ".join(f"{val} {k.upper()}" for k, val in d.items()) for d in v
            )
        return joiner.join(str(x) for x in v)
    if isinstance(v, dict):
        return ", ".join(f"{val} {k.upper()}" for k, val in v.items())
    return str(v)


def fetch_smogon(kind, fmt):  # kind = "analyses" or "sets", fmt = "gen9ou"
    url = f"https://data.pkmn.cc/{kind}/{fmt}.json"
    req = urllib.request.Request(url, headers={"User-Agent": "pkmngpt/1.0"})
    try:
        with urllib.request.urlopen(req, timeout=10) as resp:
            data = json.loads(resp.read())
            time.sleep(1)
    except urllib.error.HTTPError as e:
        if e.code == 404:
            return None
        raise

    return data


def fetch_smogon_analyses(gen_tier, out_dir=ANALYSES_DIR):
    """Fetch each format's analyses JSON and save to disk — run this ONCE
    (or when refreshing). Chunking then reads from disk, no re-fetch."""
    out_dir.mkdir(parents=True, exist_ok=True)
    for generation, tier in gen_tier:
        fmt = f"{generation}{tier}"
        out = out_dir / f"{fmt}.json"
        if out.exists():  # resumable — skip already-fetched
            print(f"  skip {fmt} (exists)")
            continue
        data = fetch_smogon("analyses", fmt)
        if data is not None:
            out.write_text(json.dumps(data), encoding="utf-8")
            print(f"  saved {fmt}: {len(data)} pokemon")
        time.sleep(1)
    print(f"analyses saved to {out_dir}")


def format_sets(set_data: dict) -> str:
    if not set_data:
        return "No competitive sets found."
    lines = []
    for key, s in set_data.items():
        moves = ", ".join(
            m if isinstance(m, str) else " / ".join(m) for m in s.get("moves", [])
        )
        lines.append(
            f"{s['pokemon'].title()} — {s['set_name']} ({s['generation']} {s['tier'].upper()}): "
            f"item {_opt(s.get('item'))}, ability {_opt(s.get('ability'))}, "
            f"{_opt(s.get('nature'))} nature, EVs {_opt(s.get('evs'))}, "
            f"Tera {_opt(s.get('teratypes'))}. Moves: {moves}."
        )
    return "\n".join(lines)


_splitter = RecursiveCharacterTextSplitter(
    chunk_size=config.CHUNK_SIZE,
    chunk_overlap=config.CHUNK_OVERLAP,
    separators=["\n\n", "\n", ". ", " ", ""],
    length_function=len,
)


def _pack(text, chunk_size=None, overlap=None):
    """Split cleaned prose into overlapping chunks on natural boundaries."""
    text = text.strip()
    if not text:
        return []
    return _splitter.split_text(text)


def chunk_one_smogon_analysis(analyses, generation, tier, chunk_size, overlap):
    records = []
    idx = 0

    def add_record(section, pokemon, prose):
        nonlocal idx
        prose = _clean_html(prose)
        if len(prose) < 30:
            return
        for chunk in _pack(prose, chunk_size=chunk_size, overlap=overlap):
            records.append(
                {
                    "text": chunk,
                    "pokemon_name": pokemon.lower(),
                    "topic": None,
                    "section": section,
                    "generation": generation,
                    "tier": tier,
                    "doc_type": "smogon",
                    "source": f"smogon/{generation}/{tier}/{pokemon.lower()}",
                    "chunk_index": idx,
                    "id": f"smogon/{generation}/{tier}/{pokemon.lower()}::{idx}",
                }
            )
            idx += 1

    for pokemon, analysis in analyses.items():
        if isinstance(analysis.get("overview"), str):
            add_record("Overview", pokemon, analysis["overview"])
        for setname, setdata in analysis.get("sets", {}).items():
            if isinstance(setdata, dict) and setdata.get("description"):
                add_record(setname, pokemon, setdata["description"])
        if isinstance(analysis.get("checksAndCounters"), str):
            add_record("Checks and Counters", pokemon, analysis["checksAndCounters"])

    return records


def chunk_smogon_analyses(analyses_dir=ANALYSES_DIR):
    """Chunk every saved analyses file. No network — reads from disk."""
    records = []
    for path in sorted(analyses_dir.glob("*.json")):
        fmt = path.stem  # 'gen9ou'
        generation, tier = (
            (fmt[:-2], fmt[-2:]) if fmt[-1] != "s" else (fmt[:-5], fmt[-5:])
        )  # reuse your gen/tier parser
        analyses = json.loads(path.read_text(encoding="utf-8"))
        records.extend(
            chunk_one_smogon_analysis(
                analyses, generation, tier, config.CHUNK_SIZE, config.CHUNK_OVERLAP
            )
        )
    return records


def chunk_smogon_sets(gen_tier, out_path=SETS_STORE):
    store = {}

    for generation, tier in gen_tier:
        sets_dict = fetch_smogon("sets", f"{generation}{tier}")
        if sets_dict is not None:
            for species, data in sets_dict.items():
                for set_name, moveset in data.items():
                    key = f"{generation}/{tier}/{species.lower()}/{set_name}"
                    store[key] = {
                        "pokemon": species.lower(),
                        "generation": generation,
                        "tier": tier,
                        "set_name": set_name,
                        "moves": moveset.get("moves", []),
                        "ability": moveset.get("ability"),
                        "item": moveset.get("item"),
                        "nature": moveset.get("nature"),
                        "evs": moveset.get("evs", {}),
                        "ivs": moveset.get("ivs", {}),
                        "teratypes": moveset.get("teratypes", []),
                        "level": moveset.get("level"),
                    }
        print(f"{generation}{tier} parsed")
    print(f"{len(store)} sets")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(
        json.dumps(store, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(f"\nWrote {len(store)} sets to {out_path}")
    return store


def load_set_store(path: Path = SETS_STORE) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def get_sets(store, pokemon, generation=None, tier=None):
    """All sets for a Pokemon, optionally filtered to a gen/tier."""
    pk = pokemon.lower()
    return {
        k: v
        for k, v in store.items()
        if v["pokemon"] == pk
        and (generation is None or v["generation"] == generation)
        and (tier is None or v["tier"] == tier)
    }


if __name__ == "__main__":
    gen_tier = [
        (f"gen{gen}", tier)
        for tier in ("ubers", "ou", "uu", "ru", "nu", "pu", "zu")
        for gen in set(range(1,10)) | {"9champions"}
    ]

    chunk_smogon_sets(gen_tier)
    fetch_smogon_analyses(gen_tier)
