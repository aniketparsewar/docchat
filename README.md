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


**Chunk size trades precision against recall; the question shape decides.**
200-token chunks scored higher on point lookups (0.49 vs 0.47) — pure chunks
align sharply with focused questions. But they scattered the Amazon
transactions across chunks, so "list all" failed. No universal winner.

**The min-score threshold is coupled to the chunking.** store-200 needed
0.367, store-500 needed 0.328, store-1000 (a single chunk) had no separable
threshold at all. A tuned threshold doesn't survive re-chunking — the
prompt-level refusal is the real safety net, min-score is the backstop.

**Retrieval recall ≠ answer recall.** Hybrid retrieval (Reciprocal Rank Fusion
of vector + keyword rankings) provably surfaced all three Amazon transactions
in the top-3 chunks — and the model still dropped one in generation, 3/3
times. Fixing the index wasn't enough; an explicit enumeration instruction in
the system prompt fixed it. Measure the answer, not just the retrieval.

**Evals need repeats.** Single runs were noisy: the same store passed
merchant-listing one run and failed the next. `--repeat 3` with majority vote
separates solid passes (3/3), solid failures (0/3), and flaky tests — and the
stability table tells you which is which.

**LLM judges fail three ways.** Noise (flaky verdicts on identical-quality
answers), systematic misreading (failing an answer the rubric explicitly
passes), and hallucinated verdicts (claiming a listed transaction is missing).
Rule: mechanical checks where truth is mechanical, judges only where
subjectivity is unavoidable.


## Exercises completed

- [x] Chunk-size experiment measured with the prompt-lab harness
- [x] Paragraph-aware chunker
- [x] min_score tuned on answerable vs unanswerable questions
- [x] Hybrid keyword + vector retrieval
- [ ] STRETCH: swapped numpy store for ChromaDB
