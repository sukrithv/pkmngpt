"""Retrieval sanity check — run real questions, read what comes back by hand."""

from src.rag_llm.vectorstore import query

tests = [
    # does a specific Pokemon still surface among 1025, not fuzz?
    ("why is Gyarados violent", "gyarados"),
    ("what is Koraidon's true form", "koraidon"),
    # concept still works
    ("how does the damage formula work", None),
    ("what is Mega Evolution", None),
    # THE cross-contamination probe you flagged: a Pokemon that's also a plot character
    (
        "tell me about Type: Null",
        "type-null",
    ),  # should get creature page, not Alola plot
    ("what happens in the Sun and Moon story", None),  # should get region/plot
]
for q, name in tests:
    print(f"\nQ: {q}  (filter={name})")
    for h in query(q, top_k=3, pokemon_name=name):
        label = h.get("pokemon_name") or h.get("section") or h.get("source")
        print(f"  [{label}] {h['score']:.3f}  {h['text'][:90]}")
