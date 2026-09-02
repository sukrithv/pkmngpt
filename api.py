"""FastAPI backend that exposes your PokemonRouter to the web UI.

Run alongside your existing system:
    uv run uvicorn api:app --reload --port 8000

Then open ui.html in a browser (it POSTs to http://localhost:8000/ask).

This imports your REAL router — adjust the import paths to match your project.
"""

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pydantic import BaseModel

# --- your real system ---
from src.dspy_eval import configure_dspy
from src.router import PokemonRouter

configure_dspy()
router = PokemonRouter()

app = FastAPI(title="pkmngpt")

# allow the local HTML file / any localhost origin to call this
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # local dev only; tighten for anything shared
    allow_methods=["*"],
    allow_headers=["*"],
)

BRANCHES = [
    "sql",
    "pokemon_rag",
    "concept_rag",
    "api_tool",
    "smogon_sets",
    "smogon_analyses",
    "damage_calc",
]


class Query(BaseModel):
    question: str


@app.get("/")
def index():
    # serve the UI from the same origin so there are no CORS issues at all
    return FileResponse("ui.html")


@app.post("/ask")
def ask(q: Query):
    try:
        pred = router(question=q.question)
    except Exception as e:
        return {
            "answer": f"Something went wrong answering that: {e}",
            "routing": {},
            "error": True,
        }

    routing = pred.routing if hasattr(pred, "routing") else None
    fired = {}
    if routing is not None:
        for b in BRANCHES:
            fired[b] = bool(getattr(routing, f"needs_{b}", False))

    meta = {}
    if routing is not None:
        meta = {
            "pokemon_name": getattr(routing, "pokemon_name", "") or "",
            "generation": getattr(routing, "generation", "") or "",
            "tier": getattr(routing, "tier", "") or "",
        }

    return {
        "answer": pred.answer,
        "routing": fired,
        "meta": meta,
        "error": False,
    }
