"""One-shot pipeline: chunk everything in data/raw/ and (re)build the vector index.

Run from the project root:
    python scripts/build_index.py
"""

import sys
from pathlib import Path

from src.rag_llm import ingest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.rag_llm import vectorstore


def main():
    chunks = ingest.build_chunks()
    if not chunks:
        print(
            "No documents found in data/raw/. Add .txt/.md/.pdf/.json files and re-run."
        )
        return
    ingest.write_chunks(chunks)
    print(f"Chunked {len(chunks)} passages.")
    vectorstore.build_index()


if __name__ == "__main__":
    main()
