# DocChat — RAG: chat with your PDFs

Retrieval-Augmented Generation, hand-rolled: chunk the document, embed the
chunks, store the vectors, and at question time retrieve the most relevant
chunks and answer with citations.

```bash
python ingest.py --pdf statement.pdf --store ./store
python chat.py --store ./store
```

Project 3 Builds on project 1 (PDF
extraction) and is evaluated by project 2's harness (prompt-lab).

## How it works

```
ingest.py:  PDF -> text -> overlapping token chunks -> embeddings -> store/
chat.py:    question -> embed -> cosine-similarity search -> top-k chunks
            -> LLM answers using ONLY those chunks, citing [1], [2], ...
```

The vector store (`src/store.py`) is deliberately hand-written numpy, not a
vector database — a vector DB is, at its core, a matrix plus cosine
similarity. The `min_score` threshold makes the pipeline refuse when nothing
relevant is retrieved, instead of improvising.

## What I learned

- Embeddings: what they are and why similar meanings land near each other
- Chunking: size/overlap tradeoffs and why they matter for retrieval
- Cosine similarity as the retrieval workhorse
- Grounded generation: citations + refusal threshold as hallucination defenses
- Cost anatomy of RAG: one-time embedding cost vs per-question cost

## Exercises completed

- [x] Chunk-size experiment measured with the prompt-lab harness
- [ ] Paragraph-aware chunker
- [ ] min_score tuned on answerable vs unanswerable questions
- [ ] Hybrid keyword + vector retrieval
- [ ] STRETCH: swapped numpy store for ChromaDB
