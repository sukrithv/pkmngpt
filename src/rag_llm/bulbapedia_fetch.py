"""Fetch Bulbapedia articles as clean wikitext via the MediaWiki API, and chunk
them for RAG.

Hits the structured backend (action=parse) instead of scraping rendered HTML,
so section structure (== headers ==) and templates survive intact for the
chunker. Pokemon articles go to data/raw/bulbapedia/; concept/mechanics
articles to data/raw/concepts/.
"""

import json
import re
import time
import urllib.parse
import urllib.request
from pathlib import Path

import mwparserfromhell
from langchain_text_splitters import RecursiveCharacterTextSplitter

import config

BULBA_API = "https://bulbapedia.bulbagarden.net/w/api.php"
POKEMON_DIR = config.DATA_RAW_DIR / "bulbapedia" / "pokemon"
CONCEPT_DIR = config.DATA_RAW_DIR / "bulbapedia" / "concepts"
ARTICLES_JSON = config.ROOT_DIR / "data" / "full_article_list.json"
HEADERS = {
    "User-Agent": (
        "pkmngpt/1.0 "
        "(https://github.com/sukrithv/pkmngpt; sukrithv@gmail.com) "
        "python-urllib/3.12"
    )
}
SLEEP = 1.0

# Pokemon articles: everything from these sections down is mechanical (SQL owns it).
DROP_FROM_SECTIONS = {
    "game data",
    "stats",
    "learnset",
    "in other languages",
    "external links",
    "references",
    "trivia",
}

# Concept articles: stop at the first non-prose / big-roster / boilerplate section.
CONCEPT_DROP_FROM = {
    "trivia",
    "list of",
    "gallery",
    "references",
    "notes",
    "in other languages",
    "external links",
    "see also",
    "in the anime",
    "in the manga",
    "in the tcg",
    "related articles",
    "in the spin-off games",  # keep core-series, drop spin-off detail
}

# Inline content templates: which arg holds the display text.
_CONTENT_TEMPLATES = {
    "p": "last",
    "m": "last",
    "a": "last",
    "i": "last",
    "type": "last",
    "wp": "last",
    "game": "last",
    "game2": "last",
    "dl": "last",
    "pkmn": "last",
    "obp": "first",
    "tt": "first",
    # unlisted templates (aniseries|GS etc.) carry no useful display text -> dropped
}

TITLE_ALIASES = {
    "mr-mime": "Mr. Mime (Pokémon)",
    "ho-oh": "Ho-Oh (Pokémon)",
    "mime-jr": "Mime Jr. (Pokémon)",
    "porygon-z": "Porygon-Z (Pokémon)",
    "type-null": "Type: Null (Pokémon)",
    "jangmo-o": "Jangmo-o (Pokémon)",
    "hakamo-o": "Hakamo-o (Pokémon)",
    "kommo-o": "Kommo-o (Pokémon)",
    "sirfetchd": "Sirfetch'd (Pokémon)",
    "mr-rime": "Mr. Rime (Pokémon)",
    "wo-chien": "Wo-Chien (Pokémon)",
    "chien-pao": "Chien-Pao (Pokémon)",
    "ting-lu": "Ting-Lu (Pokémon)",
    "chi-yu": "Chi-Yu (Pokémon)",
}


# --------------------------------------------------------------------------- #
# Fetching
# --------------------------------------------------------------------------- #
misses = []


def _api_get(params: dict):
    params = {**params, "format": "json"}
    url = f"{BULBA_API}?{urllib.parse.urlencode(params)}"
    req = urllib.request.Request(url, headers=HEADERS)
    with urllib.request.urlopen(req, timeout=20) as resp:
        return json.loads(resp.read())


def fetch_wikitext(title: str, retries=3) -> str | None:
    for attempt in range(retries):
        try:
            data = _api_get({"action": "parse", "page": title, "prop": "wikitext"})
            if "error" in data:
                return None  # genuine "page missing" — don't retry
            return data["parse"]["wikitext"]["*"]
        except (urllib.error.URLError, TimeoutError):
            if attempt < retries - 1:
                time.sleep(2**attempt)  # 1s, 2s, 4s backoff
                continue
            print(f"  TIMEOUT (gave up): {title}")
            return None


def pokemon_page_title(name: str) -> str:
    if name in TITLE_ALIASES:
        return TITLE_ALIASES[name]
    return f"{name.replace('-', ' ').title()} (Pokémon)"


def fetch_pokemon(name: str) -> Path | None:
    """Fetch one Pokemon's article into data/raw/bulbapedia/pokemon."""
    out = POKEMON_DIR / f"{name}.wikitext"
    if out.exists():
        return out
    title = pokemon_page_title(name)
    wikitext = fetch_wikitext(title)
    if wikitext is None:
        print(f"  MISS: {title}")
        misses.append(title)
        return None
    POKEMON_DIR.mkdir(parents=True, exist_ok=True)
    out.write_text(wikitext, encoding="utf-8")
    print(f"  OK:   {title} ({len(wikitext)} chars)")
    return out


def fetch_article(title: str, out_name: str | None = None) -> Path | None:
    """Fetch a page by exact title (no '(Pokémon)' suffix) into data/raw/bulbapedia/concepts/,
    for concept pages like 'Mega Evolution', 'Breeding', 'Nature'."""

    wikitext = fetch_wikitext(title)
    if wikitext is None:
        print(f"  MISS: {title}")
        misses.append(title)
        return None
    CONCEPT_DIR.mkdir(parents=True, exist_ok=True)
    slug = out_name or title.lower().replace(" ", "-")
    out = CONCEPT_DIR / f"{slug}.wikitext"
    if out.exists():
        return out
    out.write_text(wikitext, encoding="utf-8")
    print(f"  OK:   {title} ({len(wikitext)} chars)")
    return out


def fetch_many(names: list[str]):
    """Bulk-fetch Pokemon articles, politely (1 req/sec)."""
    POKEMON_DIR.mkdir(parents=True, exist_ok=True)
    fetched = []
    for name in names:
        try:
            path = fetch_pokemon(name)
            if path:
                fetched.append(path)
        except Exception as e:
            print(f"  ERROR fetching {name}: {e}")
        time.sleep(SLEEP)
    print(f"\nFetched {len(fetched)}/{len(names)} articles into {POKEMON_DIR}")
    return fetched


def fetch_many_articles(titles: list[str]):
    """Bulk-fetch concept articles by exact title, politely."""
    CONCEPT_DIR.mkdir(parents=True, exist_ok=True)
    fetched = []
    for title in titles:
        try:
            path = fetch_article(title)
            if path:
                fetched.append(path)
        except Exception as e:
            print(f"  ERROR fetching {title}: {e}")
        time.sleep(SLEEP)
    print(f"\nFetched {len(fetched)}/{len(titles)} articles into {CONCEPT_DIR}")
    return fetched


# --------------------------------------------------------------------------- #
# Section splitting
# --------------------------------------------------------------------------- #


def split_sections(wikitext: str, drop_from=None, keep_lead=False):
    """Yield (section_title, body) for prose sections, stopping at the first
    section whose title is in drop_from. If keep_lead, also yield the text
    before the first header as ('Lead', ...)."""
    if drop_from is None:
        drop_from = DROP_FROM_SECTIONS

    headers = []
    for m in re.finditer(r"^={2,6}\s*(.+?)\s*={2,6}\s*$", wikitext, re.MULTILINE):
        headers.append((m.start(), m.end(), m.group(1).strip()))

    if keep_lead and headers and headers[0][0] > 0:
        lead = wikitext[: headers[0][0]].strip()
        if lead:
            yield "Lead", lead
    elif keep_lead and not headers:
        # whole article has no headers — treat all of it as the lead
        if wikitext.strip():
            yield "Lead", wikitext

    for idx, (start, end, title) in enumerate(headers):
        if title.lower() in drop_from or title.lower().startswith("list of"):
            break
        body_start = end
        body_end = headers[idx + 1][0] if idx + 1 < len(headers) else len(wikitext)
        yield title, wikitext[body_start:body_end]


# --------------------------------------------------------------------------- #
# Cleaning
# --------------------------------------------------------------------------- #


def clean_wikitext(text: str) -> str:
    """Preserve display text of known content templates, strip everything else
    (leftover templates, tables, links, refs, comments, captions)."""
    text = re.sub(r"<!--.*?-->", "", text, flags=re.DOTALL)  # HTML comments
    text = re.sub(
        r"<ref[^>]*>.*?</ref>", "", text, flags=re.DOTALL
    )  # named/inline refs
    text = re.sub(r"<ref[^>]*/>", "", text)  # self-closing refs
    text = re.sub(r"^:''.*?''\s*$", "", text, flags=re.MULTILINE)  # disambig preamble
    text = re.sub(r"<font[^>]*>", "", text)

    code = mwparserfromhell.parse(text)
    for tmpl in list(code.filter_templates()):
        name = str(tmpl.name).strip().lower()
        rule = _CONTENT_TEMPLATES.get(name)
        if rule is None:
            continue  # unknown -> strip_code removes it
        args = [str(p.value).strip() for p in tmpl.params if not p.showkey]
        replacement = "" if not args else (args[0] if rule == "first" else args[-1])
        try:
            code.replace(tmpl, replacement)
        except ValueError:
            pass  # already removed (nested)

    text = code.strip_code(normalize=True, collapse=True)
    text = re.sub(
        r"(?im)^\s*(?:thumb|left|right|\d+px|[^\n|]*\.(?:png|jpg|jpeg|gif))\b[^\n]*$",
        "",
        text,
    )  # image/caption lines strip_code left behind
    text = re.sub(r"\n{3,}", "\n\n", text).strip()
    return text


# --------------------------------------------------------------------------- #
# Chunking
# --------------------------------------------------------------------------- #

_splitter = RecursiveCharacterTextSplitter(
    chunk_size=config.CHUNK_SIZE,
    chunk_overlap=config.CHUNK_OVERLAP,
    separators=["\n\n", "\n", ". ", " ", ""],
    length_function=len,
)


def _is_number_soup(text: str) -> bool:
    """True if the text is mostly numbers/digits — a stripped data table, not prose."""
    if not text:
        return True
    # ratio of digit+comma+space chars to total; prose is <~30%, tables >~60%
    numeric = sum(c.isdigit() or c in ", " for c in text)
    return numeric / len(text) > 0.5


def _pack(text, chunk_size=None, overlap=None):
    """Split cleaned prose into overlapping chunks on natural boundaries."""
    text = text.strip()
    if not text:
        return []
    return _splitter.split_text(text)


def chunk_bulbapedia(raw_text, pokemon_name, chunk_size, overlap):
    """Chunk a Pokemon article: skip lead (infobox junk), keep prose sections,
    stop at the first mechanical section."""
    records = []
    idx = 0
    for section_title, body in split_sections(raw_text, drop_from=DROP_FROM_SECTIONS):
        prose = clean_wikitext(body)
        if len(prose) < 40:
            continue
        for chunk in _pack(prose, chunk_size, overlap):
            records.append(
                {
                    "text": chunk,
                    "pokemon_name": pokemon_name,
                    "topic": None,
                    "section": section_title,
                    "source": f"bulbapedia/pokemon/{pokemon_name}",
                    "chunk_index": idx,
                    "id": f"bulbapedia/pokemon/{pokemon_name}::{idx}",
                }
            )
            idx += 1
    return records


def chunk_concept(raw_text, topic, chunk_size, overlap):
    """Chunk a concept article: KEEP the lead definition and prose sections,
    stop at the first data-table/boilerplate section."""
    records = []
    idx = 0
    for section_title, body in split_sections(
        raw_text, drop_from=CONCEPT_DROP_FROM, keep_lead=True
    ):
        prose = clean_wikitext(body)
        if len(prose) < 40 or _is_number_soup(prose):
            continue
        for chunk in _pack(prose, chunk_size, overlap):
            records.append(
                {
                    "text": chunk,
                    "pokemon_name": None,
                    "topic": topic,
                    "section": section_title,
                    "source": f"bulbapedia/concepts/{topic}",
                    "chunk_index": idx,
                    "id": f"bulbapedia/concepts/{topic}::{idx}",
                }
            )
            idx += 1
    return records


# --------------------------------------------------------------------------- #
# Inspection entry point
# --------------------------------------------------------------------------- #

if __name__ == "__main__":
    # with open(ARTICLES_JSON, "r", encoding="utf-8") as file:
    #     articles = json.load(file)
    # fetch_many(articles["pokemon"])
    # fetch_many_articles(articles["concepts"])

    # print(misses)

    # raw_text = (CONCEPT_DIR / "ability.wikitext").read_text(encoding="utf-8")
    # chunks = chunk_concept(raw_text, "ability", config.CHUNK_SIZE, config.CHUNK_OVERLAP)
    # print(f"\n{len(chunks)} chunks\n" + "=" * 60)
    # for c in chunks:
    #     print(f"\n[{c['section']}] ({len(c['text'])} chars)")
    #     print(c["text"][:300])

    import json
    from collections import Counter

    import config

    chunks = [json.loads(l) for l in open(config.CHUNKS_FILE)]
    print("total chunks:", len(chunks))
    per_source = Counter(c["source"] for c in chunks)
    print("0-1 chunk sources:", [s for s, n in per_source.items() if n <= 1][:20])
    print(">40 chunk sources:", [(s, n) for s, n in per_source.items() if n > 40][:10])
    print("median chunks/source:", sorted(per_source.values())[len(per_source) // 2])

    raw = open("data/raw/bulbapedia/pokemon/volbeat.wikitext").read()
    print(len(raw))
