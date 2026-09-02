"""Load raw Pokemon documents and split them into overlapping chunks.

Supports .txt, .md, .json (list of {"title"/"name", "text"} or dict of any
shape — dumped as pretty JSON text) and .pdf files dropped into data/raw/.
"""

import json
from pathlib import Path

from pypdf import PdfReader

import config
from src.rag_llm.bulbapedia_fetch import chunk_bulbapedia, chunk_concept
from src.rag_llm.smogon_fetch import chunk_smogon_analyses


def _read_txt(path):
    return path.read_text(encoding="utf-8", errors="ignore")


def _read_pdf(path):
    reader = PdfReader(str(path))
    return "\n".join(page.extract_text() or "" for page in reader.pages)


def _read_json(path):
    data = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(data, list):
        parts = []
        for item in data:
            if isinstance(item, dict):
                title = item.get("name") or item.get("title") or ""
                body = item.get("text") or item.get("description") or json.dumps(item)
                parts.append(f"{title}\n{body}".strip())
            else:
                parts.append(str(item))
        return "\n\n".join(parts)
    return json.dumps(data, indent=2)


LOADERS = {
    ".txt": _read_txt,
    ".md": _read_txt,
    ".wikitext": _read_txt,
    ".pdf": _read_pdf,
    ".json": _read_json,
}


def load_documents(raw_dir=config.DATA_RAW_DIR):
    """Yield (source_path, text) for every supported file under raw_dir."""
    for path in sorted(raw_dir.rglob("*")):
        if not path.is_file():
            continue
        loader = LOADERS.get(path.suffix.lower())
        if loader is None:
            continue
        text = loader(path).strip()
        if text:
            yield str(path.relative_to(raw_dir)), text


def chunk_text(text, chunk_size=config.CHUNK_SIZE, overlap=config.CHUNK_OVERLAP):
    """Naive fixed-size character splitter with overlap.

    Good enough as a starting point — swap in something smarter (sentence-
    aware, markdown-header-aware, etc.) once you see how retrieval quality
    looks on your real documents.
    """
    if overlap >= chunk_size:
        raise ValueError("CHUNK_OVERLAP must be smaller than CHUNK_SIZE")

    chunks = []
    start = 0
    text_len = len(text)
    while start < text_len:
        end = min(start + chunk_size, text_len)
        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)
        if end == text_len:
            break
        start = end - overlap
    return chunks


def build_chunks(raw_dir=config.DATA_RAW_DIR):
    records = []
    for source, text in load_documents(raw_dir):
        if source.startswith("bulbapedia/pokemon"):
            name = Path(source).stem
            records.extend(
                chunk_bulbapedia(text, name, config.CHUNK_SIZE, config.CHUNK_OVERLAP)
            )
        elif source.startswith("bulbapedia/concepts"):
            topic = Path(source).stem
            records.extend(
                chunk_concept(text, topic, config.CHUNK_SIZE, config.CHUNK_OVERLAP)
            )
        elif source.startswith("smogon/"):
            continue
        else:
            # fallback naive path for other doc types (.txt/.md/.pdf/.json)
            for i, chunk in enumerate(chunk_text(text)):
                records.append(
                    {
                        "id": f"{source}::{i}",
                        "source": source,
                        "pokemon_name": None,
                        "topic": None,
                        "section": None,
                        "chunk_index": i,
                        "text": chunk,
                    }
                )

    records.extend(chunk_smogon_analyses())

    # ensure every record has a stable id
    for i, r in enumerate(records):
        r.setdefault("id", f"{r['source']}::{i}")
    return records


def write_chunks(records, out_path=config.CHUNKS_FILE):
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record) + "\n")


if __name__ == "__main__":
    # chunks = build_chunks()
    # write_chunks(chunks)
    # print(f"Wrote {len(chunks)} chunks to {config.CHUNKS_FILE}")

    import json
    from collections import Counter

    import config

    chunks = [json.loads(l) for l in open(config.CHUNKS_FILE)]
    smogon = [c for c in chunks if c.get("doc_type") == "smogon"]
    print("smogon chunks:", len(smogon))
    print(
        "by format:", Counter(f"{c.get('generation')}/{c.get('tier')}" for c in smogon)
    )
    print("distinct pokemon:", len({c["pokemon_name"] for c in smogon}))
