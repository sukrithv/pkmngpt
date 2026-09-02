"""Chroma-backed vector store for the Pokemon chunk corpus.

Embeddings are computed locally with sentence-transformers so indexing never
requires an API key or a running Ollama server — only generation (the LLM
call) needs a backend chosen.
"""

import json

import chromadb
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

import config

_embedder = None
chromadb.configure(anonymized_telemetry=False)


def get_embedder():
    global _embedder
    if _embedder is None:
        _embedder = SentenceTransformer(config.EMBEDDING_MODEL)
    return _embedder


def get_collection():
    client = chromadb.PersistentClient(path=config.CHROMA_DIR)
    return client.get_or_create_collection(
        name=config.CHROMA_COLLECTION,
        metadata={"hnsw:space": "cosine"},
    )


def build_index(chunks_file=config.CHUNKS_FILE, batch_size=64, rebuild=False):
    records = [json.loads(line) for line in chunks_file.open(encoding="utf-8")]
    client = chromadb.PersistentClient(path=config.CHROMA_DIR)
    if rebuild:
        try:
            client.delete_collection(config.CHROMA_COLLECTION)
        except Exception:
            pass
    collection = client.get_or_create_collection(
        name=config.CHROMA_COLLECTION, metadata={"hnsw:space": "cosine"}
    )

    existing = set(collection.get(include=[])["ids"])  # ids already indexed
    todo = [r for r in records if r["id"] not in existing]  # only new ones
    print(f"{len(existing)} already indexed, {len(todo)} new")

    embedder = get_embedder()
    for i in tqdm(range(0, len(todo), batch_size), desc="Embedding new chunks"):
        batch = todo[i : i + batch_size]
        embeddings = embedder.encode([r["text"] for r in batch]).tolist()
        collection.upsert(
            ids=[r["id"] for r in batch],
            embeddings=embeddings,
            documents=[r["text"] for r in batch],
            metadatas=[
                {
                    "source": r["source"],
                    "pokemon_name": r.get("pokemon_name") or "",
                    "topic": r.get("topic") or "",
                    "section": r.get("section") or "",
                    "doc_type": r.get("doc_type") or "",
                    "generation": r.get("generation") or "",
                    "tier": r.get("tier") or "",
                    "chunk_index": r.get("chunk_index", 0),
                }
                for r in batch
            ],
        )
    print(f"Indexed {len(records)} chunks into collection '{config.CHROMA_COLLECTION}'")


def query(
    text,
    top_k=config.TOP_K,
    pokemon_name=None,
    doc_type=None,
    generation=None,
    tier=None,
):
    embedder = get_embedder()
    collection = get_collection()
    embedding = embedder.encode([text]).tolist()

    conditions = []
    if pokemon_name:
        conditions.append({"pokemon_name": pokemon_name})
    if doc_type:
        conditions.append({"doc_type": doc_type})
    if generation:
        conditions.append({"generation": generation})
    if tier:
        conditions.append({"tier": tier})

    if not conditions:
        where = None
    elif len(conditions) == 1:
        where = conditions[0]
    else:
        where = {"$and": conditions}  # Chroma needs $and for multiple filters

    results = collection.query(query_embeddings=embedding, n_results=top_k, where=where)
    hits = []
    for doc, meta, dist in zip(
        results["documents"][0], results["metadatas"][0], results["distances"][0]
    ):
        hits.append(
            {
                "text": doc,
                "source": meta["source"],
                "pokemon_name": meta.get("pokemon_name", ""),
                "topic": meta.get("topic", ""),
                "section": meta.get("section", ""),
                "doc_type": meta.get("doc_type", ""),
                "generation": meta.get("generation", ""),
                "tier": meta.get("tier", ""),
                "score": 1 - dist,
            }
        )
    return hits


if __name__ == "__main__":
    build_index(rebuild=True)
