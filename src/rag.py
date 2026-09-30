"""The RAG pipeline: retrieve the best chunks, then generate a cited answer.

Flow: question -> embed -> cosine search -> top-k chunks -> prompt with
numbered excerpts -> LLM answers citing [1], [2], ...

The min_score guard is a refusal mechanism: if even the best chunk is
dissimilar to the question, the document probably doesn't contain the
answer, and we say so instead of letting the model improvise.
"""
from . import embeddings, llm, store

SYSTEM_RAG = """You answer questions using ONLY the excerpts below.
Rules:
- Cite every factual claim with the excerpt number, like [1] or [2].
- If the excerpts don't contain the answer, say so plainly. Do not use
  outside knowledge for facts, though you may explain terms in plain words.
- Keep answers short."""


def answer(question: str, chunks: list[str], matrix,
           top_k: int = 3, min_score: float = 0.25) -> dict:
    qvec = embeddings.embed_one(question)
    hits = store.search(qvec, matrix, top_k=top_k)

    best_idx, best_score = hits[0]
    if best_score < min_score:
        return {
            "answer": ("I couldn't find anything relevant in the document "
                       f"(best match score {best_score:.2f})."),
            "hits": hits,
            "chunks_used": [],
            "cost_usd": 0.0,
            "refused": True,
        }

    context = "\n\n".join(
        f"[{rank}] {chunks[idx]}" for rank, (idx, _) in enumerate(hits, start=1)
    )
    user_prompt = (f"EXCERPTS:\n---\n{context}\n---\n\n"
                   f"QUESTION: {question}")
    resp = llm.ask(SYSTEM_RAG, user_prompt)
    return {
        "answer": resp["answer"],
        "hits": hits,
        "chunks_used": [idx for idx, _ in hits],
        "cost_usd": resp["cost_usd"],
        "input_tokens": resp["input_tokens"],
        "output_tokens": resp["output_tokens"],
        "refused": False,
    }


# ---- EXERCISES ----
# 1. Tune min_score: ask an answerable and an unanswerable question, look at
#    the scores, and pick a threshold that separates them.
# 2. Hybrid retrieval: pre-filter chunks by keyword overlap with the question,
#    then rank the survivors by vector similarity. When does this beat pure
#    vector search? (Hint: try it on "AMAZON.CA".)
# 3. Point your prompt-lab harness at rag.answer() as the system under test.
#    Write 5 golden Q&As for your statement and measure the pipeline.
