# pkmngpt — a multi-engine Pokemon Q&A assistant

A question-answering system for Pokemon that routes each question to
whichever engine can actually answer it — SQL for facts, RAG for lore and
competitive analysis, a competitive-set lookup, or a real damage calculator —
then synthesizes the results into one answer. Generation is a config flag:
local **Ollama** or the **Claude API**.

Ask it from a chat UI, a FastAPI endpoint, or the command line.

## How a question gets answered

```
question
    │
    ▼  src/router.py  PokemonRouter
    │     a DSPy classifier (RouterClassifier) reads the question and sets
    │     one or more flags: needs_sql / needs_pokemon_rag / needs_concept_rag /
    │     needs_api_tool / needs_smogon_sets / needs_smogon_analyses /
    │     needs_damage_calc — plus extracted pokemon_name / generation / tier
    │
    ├─▶ needs_sql            → src/sql_llm/sql_agent.py   TextToSQL over pokeapi.db
    ├─▶ needs_pokemon_rag    → src/rag_llm/rag.py         RAGAnswer over Bulbapedia (Chroma)
    ├─▶ needs_concept_rag    → src/rag_llm/rag.py         RAGAnswer over game-mechanic articles
    ├─▶ needs_smogon_sets    → src/rag_llm/smogon_fetch.py get_sets() from the set store
    ├─▶ needs_smogon_analyses→ src/rag_llm/rag.py         RAGAnswer over Smogon analyses
    └─▶ needs_damage_calc    → src/smogon_calc/calc_llm.py DamageCalcModule
                                  (extracts a scenario, calls the @smogon/calc
                                   Node bridge, formats a damage range + KO verdict)
    │
    ▼  evidence from every branch that fired is merged and labeled by source
    ▼  src/router.py  SynthesizeAnswer (DSPy) turns it into one grounded answer
    │
 answer + which engines fired + extracted routing metadata
```

`src/llm.py` is the swappable generation backend (`LLM_BACKEND=ollama` or
`claude` in `.env`) that DSPy calls under the hood — every engine above talks
to it the same way.

## The engines

| Engine | Data source | Module |
|---|---|---|
| **Text-to-SQL** | `pokeapi.db`, a local SQLite build of PokeAPI's relational data (species, stats, types, moves, abilities, evolution, ...) | `src/sql_llm/sql_agent.py`, `src/sql_llm/sql_pokeapi.py` |
| **Pokemon / concept RAG** | Bulbapedia articles (creature pages + mechanic/lore pages), chunked and embedded into Chroma | `src/rag_llm/rag.py`, `src/rag_llm/vectorstore.py` |
| **Smogon sets** | A keyed JSON store of competitive movesets (`gen/tier/pokemon/setname`) | `src/rag_llm/smogon_fetch.py` |
| **Smogon analyses (RAG)** | Smogon competitive-viability writeups, chunked into the same Chroma index, filtered by `doc_type`/`generation`/`tier` | `src/rag_llm/rag.py` + `smogon_fetch.py`'s analysis chunker |
| **Damage calculator** | `@smogon/calc` (the JS library behind Smogon's own calculator), reached via a small Express bridge | `src/smogon_calc/client.py`, `set_adapter.py`, `calc_llm.py`, `src/smogon_calc/node/server.js` |

The router extracts a scenario from natural language (attacker/defender,
move, item, ability, weather, boosts, generation), fills in a competitive set
from the Smogon set store when one isn't specified, and calls the Node
service for the actual computation — because the damage formula's edge cases
(abilities, items, weather, terrain, generation differences) are exactly
what `@smogon/calc` already gets right.

## Interfaces

- **Chat UI** — `ui.html`, a single static page that POSTs to the API and
  color-codes which engine(s) answered.
- **API** — `api.py`, a FastAPI app (`POST /ask`) that wraps `PokemonRouter`
  and also serves `ui.html` at `/`.
- **CLI** — `scripts/ask.py`, a one-shot question against the Pokemon/concept
  RAG engine only (not the full router).

## Setup

```bash
python -m venv .venv && source .venv/bin/activate   # or: uv sync
pip install -e ".[dev]"
cp .env.example .env   # then edit LLM_BACKEND and any keys

# If using Ollama: make sure the daemon is running and the model is pulled.
#   ollama pull llama3.1
# If using Claude: set ANTHROPIC_API_KEY in .env.
```

`data/`, `pokeapi.db`, and `chroma_db/` are gitignored — rebuild them once
(see below) before asking questions.

### Rebuilding the data

Each dataset is a one-time build/fetch step, then reused:

```bash
# 1. SQL data: get PokeAPI's CSV export, then build the SQLite DB.
#    (the CSVs are PokeAPI's own data/v2/csv/ directory from
#    github.com/PokeAPI/pokeapi — drop them in data/pokeapi_data/)
python -c "from src.sql_llm.sql_pokeapi import build_database; build_database('data/pokeapi_data')"

# 2. Bulbapedia + Smogon content: fetch, then chunk + embed into Chroma.
python -m src.rag_llm.bulbapedia_fetch
python -m src.rag_llm.smogon_fetch
python scripts/build_index.py

# 3. Damage calc bridge: install and start the Node service (separate terminal).
cd src/smogon_calc/node && npm install && node server.js
```

### Running it

```bash
# API + chat UI
uv run uvicorn api:app --reload --port 8000
# then open http://localhost:8000 (serves ui.html)

# CLI (Pokemon/concept RAG only)
python scripts/ask.py "Why is Gyarados so violent?"
```

## Evaluation

Three separate eval harnesses, one per concern:

| What | Run it | Data |
|---|---|---|
| Routing accuracy — did the classifier set the right flags/pokemon/gen/tier? | `python -m src.router_eval` | `eval/routing_eval_set.json` |
| SQL correctness — exact-match the query result against a gold answer | `python -m src.sql_llm.diagnose_sql` | `eval/simple_eval_set.json`, `eval/complex_eval_set.json` |
| SQL agent prompt optimization — `dspy.BootstrapFewShotWithRandomSearch` against the SQL result metric | `python -m src.optimize_react_agent` (misnamed — optimizes `TextToSQL`, not the ReAct agent below) | same as above |

`src/dspy_eval.py` (`python -m src.dspy_eval keyword`/`judge`) and
`eval/eval_set.json` are older and evaluate a different, standalone
retrieval pipeline — see "Earlier/alternate pipelines" below.

## Earlier / alternate pipelines (not part of the router)

Two things in `src/` predate or sit alongside the router above and aren't
wired into `api.py` or `PokemonRouter`. Kept in the tree, worth knowing
about before assuming everything under `src/` is on the request path:

- **`src/pokeapi.py`** — the project's original retrieval approach: live
  PokeAPI REST lookups by name-matching against hardcoded slug lists
  (`GEN_1_NAMES`, `TYPE_NAMES`, etc.), no local index. `src/dspy_eval.py`'s
  `PokemonRAG` module and its eval (`eval/eval_set.json`) still run against
  this path, but the router uses the Chroma-backed `src/rag_llm/rag.py`
  instead.
- **`src/react_agent.py`** — a `dspy.ReAct` agent that answers broader
  questions ("which fire types learn flying moves?") by calling
  `src/pokeapi.py`'s functions as tools, iterating up to `MAX_ITERS` times.
  Not imported by the router and has no test coverage; an alternate design
  for cross-entity questions that the router's SQL branch now covers for
  most cases.

## File map

| Path | Purpose |
|---|---|
| `config.py` | Central config, reads `.env` |
| `api.py` | FastAPI app: `POST /ask` → `PokemonRouter`, serves `ui.html` |
| `ui.html` | Static chat UI |
| `src/router.py` | `PokemonRouter` — classifies, dispatches to engines, synthesizes |
| `src/router_demos.py` | Few-shot demos for the router's classifier |
| `src/router_eval.py` | Routing accuracy eval |
| `src/llm.py` | Ollama/Claude generation, selected by `LLM_BACKEND` |
| `src/sql_llm/sql_pokeapi.py` | Schema doc + `build_database()` + query execution against `pokeapi.db` |
| `src/sql_llm/sql_agent.py` | `TextToSQL` DSPy module |
| `src/sql_llm/diagnose_sql.py` | Per-category SQL eval (pass / SQL error / wrong result / no SQL) |
| `src/rag_llm/bulbapedia_fetch.py` | Fetch + chunk Bulbapedia articles |
| `src/rag_llm/smogon_fetch.py` | Fetch/chunk Smogon analyses; the competitive set store (`get_sets`, `format_sets`) |
| `src/rag_llm/ingest.py` | Load + chunk any `data/raw/` document (routes to the fetchers above) |
| `src/rag_llm/vectorstore.py` | Embed chunks (sentence-transformers), build/query the Chroma index |
| `src/rag_llm/rag.py` | `RAGAnswer` DSPy module — retrieve + generate a grounded answer |
| `src/rag_llm/inspect_rag.py` | Manual retrieval sanity check |
| `src/smogon_calc/client.py` | Python client for the Node `@smogon/calc` bridge |
| `src/smogon_calc/set_adapter.py` | Adapts a Smogon set-store entry into the calc client's input shape |
| `src/smogon_calc/calc_llm.py` | `DamageCalcModule` — NL scenario → structured calc → formatted result |
| `src/smogon_calc/node/server.js` | Express service wrapping `@smogon/calc` |
| `src/pokeapi.py` | Live PokeAPI lookups — original retrieval path, see "Earlier/alternate pipelines" |
| `src/dspy_eval.py` | Standalone eval harness for `src/pokeapi.py`'s retrieval (legacy) |
| `src/react_agent.py`, `src/optimize_react_agent.py` | ReAct broad-question agent + SQL-agent prompt optimizer, see above |
| `scripts/ask.py` | CLI: ask the Pokemon/concept RAG engine a question |
| `scripts/build_index.py` | CLI: chunk `data/raw/` + (re)build the Chroma index |
| `eval/` | Eval sets — routing, SQL (simple/complex), and the legacy RAG set |
| `data/full_article_list.json` | Known Pokemon/concept name lists the router resolves against |
| `next_phases_plan.md` | Design notes for the damage calculator and a planned DQN self-play battler |
| `extension.md` | Design notes for DSPy prompt optimization vs. local fine-tuning, per backend |

## Known issue

`scripts/ask.py` calls `rag.answer(...)`; that wrapper now exists on
`src/rag_llm/rag.py` but the local dev environment currently hits an
unrelated `numpy`/`dspy` import error (`TypeError: data type 'bool' not
understood`) when `chromadb` is imported, likely a version mismatch between
pinned dependencies. Worth resolving (e.g. re-pinning `numpy`/`chromadb`
versions and re-locking) before relying on the RAG-backed CLI or router.

## Not yet done

- A real eval set for the router's synthesis step (only routing accuracy and
  SQL correctness are measured today, not final-answer quality end to end).
- DSPy prompt optimization for the router's classifier and RAG generation
  (only the SQL agent has an optimizer today, in
  `src/optimize_react_agent.py`).
- The damage-calc → DQN self-play battler described in
  `next_phases_plan.md` — a separate, much larger project (needs a
  Showdown/`poke-env` battle simulator and GPU time).
- Fine-tuning the local Ollama model, per `extension.md`.
