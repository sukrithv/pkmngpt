"""Ask the RAG pipeline a question from the command line.

python scripts/ask.py "What type is Charizard?"
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.rag_llm import rag


def main():
    if len(sys.argv) < 2:
        print('Usage: python scripts/ask.py "<question>"')
        sys.exit(1)

    question = " ".join(sys.argv[1:])
    result = rag.answer(question)

    print(f"\nQ: {result['question']}\n")
    print(f"A: {result['answer']}\n")
    print("Sources:")
    for c in result["contexts"]:
        print(f"  - {c['source']} (score={c['score']:.3f})")


if __name__ == "__main__":
    main()
