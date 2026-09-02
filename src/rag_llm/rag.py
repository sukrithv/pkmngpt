"""RAG generation: retrieve relevant chunks, generate a grounded answer.

Closes the loop from retrieval (vectorstore.query) to a natural-language
answer grounded in the retrieved passages.
"""

import dspy  # dspy last

from src.dspy_eval import configure_dspy
from src.rag_llm.vectorstore import query


class GroundedAnswer(dspy.Signature):
    """Answer the question using ONLY the provided context passages. If the
    context doesn't contain the answer, say so — do not use outside knowledge.
    Be concise and factual. Cite which Pokemon or topic each fact comes from
    when relevant."""

    question: str = dspy.InputField()
    context: str = dspy.InputField(
        desc="Retrieved passages, each tagged with its source"
    )
    answer: str = dspy.OutputField()


def _format_context(hits) -> str:
    """Render retrieved chunks into a labeled context block for the LLM."""
    blocks = []
    for i, hit in enumerate(hits, 1):
        label = hit.get("topic") or hit.get("pokemon_name") or hit.get("source", "?")
        section = hit.get("section") or ""
        header = f"[{i}] {label}" + (f" — {section}" if section else "")
        blocks.append(f"{header}\n{hit['text']}")
    return "\n\n".join(blocks)


class RAGAnswer(dspy.Module):
    """Question -> retrieve top-k chunks -> generate a grounded answer."""

    def __init__(self, top_k: int = 5):
        self.top_k = top_k
        self.generate = dspy.ChainOfThought(GroundedAnswer)

    def forward(
        self,
        question: str,
        pokemon_name: str | None = None,
        doc_type: str | None = None,
        generation: str | None = None,
        tier: str | None = None,
    ) -> dspy.Prediction:
        hits = query(
            question,
            top_k=self.top_k,
            pokemon_name=pokemon_name,
            doc_type=doc_type,
            generation=generation,
            tier=tier,
        )
        if not hits:
            return dspy.Prediction(
                answer="I couldn't find any relevant information to answer that.",
                context="",
                hits=[],
            )
        context = _format_context(hits)
        result = self.generate(question=question, context=context)
        return dspy.Prediction(answer=result.answer, context=context, hits=hits)


def answer(question: str, top_k: int = 5) -> dict:
    """One-shot convenience wrapper for CLI use (scripts/ask.py)."""
    configure_dspy()
    pred = RAGAnswer(top_k=top_k)(question=question)
    return {"question": question, "answer": pred.answer, "contexts": pred.hits}


if __name__ == "__main__":
    configure_dspy()
    rag = RAGAnswer(top_k=5)

    for q in [
        "What is Mega Evolution?",
        "How is a Pokemon's friendship increased?",
        "Why is Gyarados so violent?",
        "What happens when Gengar Mega Evolves?",
    ]:
        print(f"\n{'=' * 70}\nQ: {q}\n{'=' * 70}")
        pred = rag(question=q)
        print(pred.answer)
        print(f"\n(grounded in {len(pred.hits)} passages)")
