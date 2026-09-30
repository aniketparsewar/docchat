# Project 3 Brief — DocChat (RAG over PDFs)

## Goal
Build retrieval-augmented generation from scratch: ingest a PDF into a vector
store, then chat with it — answers grounded in retrieved chunks, with
citations and a refusal threshold.

## Why this matters
RAG is the most deployed LLM architecture in industry. After this project you
can explain every piece from first principles: what an embedding is, why you
chunk, what cosine similarity does, and why the vector DB is not magic.

## Scope (keep it small)
- Local CLI + REPL. Hand-rolled numpy store (no ChromaDB yet — that's a stretch).
- One PDF at a time. No web UI, no chat history (each question is independent).

## Weekend task list

### Day 1 — Make it run
- [ ] Setup: venv, requirements, `.env` (same key as before)
- [ ] `python ingest.py --pdf <your.pdf> --store ./store` — watch the stages print
- [ ] Read `src/chunker.py`, `src/embeddings.py`, `src/store.py` until you can
      explain: why chunk, what a vector is, what cosine similarity measures
- [ ] `python chat.py --store ./store` — ask 5 questions: 3 answerable,
      2 unanswerable. Watch the similarity scores and the refusal kick in
- [ ] Exercises in `src/chunker.py` (try 200 vs 1000 token chunks — feel the
      difference before measuring it)

### Day 2 — Make it yours
- [ ] Tune `--min-score`: find the threshold that separates your answerable
      from unanswerable questions
- [ ] The killer exercise: point your **prompt-lab harness** at this pipeline.
      Write 5 golden Q&As for your statement, wrap `rag.answer()` as the system
      under test, and measure chunk-size variants against each other. Your
      project 2 now evaluates your project 3 — that's a portfolio story.
- [ ] Exercises in `src/rag.py`: hybrid retrieval, paragraph-aware chunking
- [ ] `git init`, commit, push to GitHub. Fill in README's "What I learned."

## Done means
- [ ] Ingest + chat run clean; you can explain embeddings/chunking/cosine similarity cold
- [ ] The refusal threshold demonstrably fires on unanswerable questions
- [ ] Prompt-lab scores for at least two chunk-size variants (measured, not vibes)
- [ ] Repo public on GitHub

## Stretch
- Swap `src/store.py` internals for ChromaDB (keep the function signatures)
- Multi-PDF: ingest a folder, tag chunks by source file, cite the file
- Conversation memory: include recent Q&A in the prompt (watch the token bill)

## Concepts this teaches (for interviews later)
"How does RAG work?" → chunk, embed, retrieve by similarity, generate grounded
in the chunks. "Why not just stuff the PDF in the prompt?" → context limits,
cost per question, and the lost-in-the-middle problem: LLMs reason worse over
long contexts, so retrieving the relevant slice beats sending everything.
"What are its failure modes?" → bad chunks (split tables), retrieval misses
(the answer exists but wasn't top-k), and confident answers from weak matches
(which min_score guards against).
