#!/usr/bin/env python3
"""Find a --min-score threshold that separates answerable from unanswerable questions.

For each labeled question it embeds once (batched), takes the top retrieval
score, and reports whether a clean separating threshold exists.

Usage:
    python tune_threshold.py --store ./store-200
    python tune_threshold.py --store ./store-500 --questions my_questions.json

my_questions.json format:
    [{"label": "answerable", "question": "..."}, {"label": "unanswerable", "question": "..."}]
"""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from src import embeddings, store

DEFAULT_QUESTIONS = [
    # Answerable: facts genuinely in the sample statement
    {"label": "answerable", "question": "What is my new balance?"},
    {"label": "answerable", "question": "Show all Amazon transactions."},
    {"label": "answerable", "question": "What is the APR on purchases?"},
    {"label": "answerable", "question": "What does 'balance subject to rate' mean?"},
    {"label": "answerable", "question": "How many reward points did I earn this period?"},
    {"label": "answerable", "question": "When is the payment due?"},
    # Unanswerable: plausible questions the document cannot answer
    {"label": "unanswerable", "question": "Am I eligible for a credit limit increase?"},
    {"label": "unanswerable", "question": "Does this card offer travel insurance?"},
    {"label": "unanswerable", "question": "What is the bank's CEO's name?"},
    {"label": "unanswerable", "question": "Can I add an authorized user to this account?"},
]


def suggest_threshold(scored):
    """Return (threshold, separable, detail) from [(label, top_score)] rows."""
    ans = sorted(s for l, s in scored if l == "answerable")
    una = sorted(s for l, s in scored if l == "unanswerable")
    if not ans or not una:
        return None, False, "need at least one of each label"
    worst_answerable = ans[0]      # lowest top-score among answerable
    best_unanswerable = una[-1]    # highest top-score among unanswerable
    if best_unanswerable < worst_answerable:
        mid = (best_unanswerable + worst_answerable) / 2
        return mid, True, (
            f"highest unanswerable top-score {best_unanswerable:.3f} < "
            f"lowest answerable top-score {worst_answerable:.3f}")
    overlap = [s for l, s in scored
               if (l == "unanswerable" and s >= worst_answerable)
               or (l == "answerable" and s <= best_unanswerable)]
    return None, False, (
        f"classes overlap ({len(overlap)} questions in the overlap zone): "
        f"unanswerable reaches {best_unanswerable:.3f}, "
        f"answerable drops to {worst_answerable:.3f}")


def main():
    parser = argparse.ArgumentParser(
        description="Tune --min-score on labeled answerable/unanswerable questions.")
    parser.add_argument("--store", required=True, help="Vector store directory")
    parser.add_argument("--questions", default=None,
                        help="JSON file with [{label, question}] (default: built-in set)")
    parser.add_argument("--top-k", type=int, default=3)
    args = parser.parse_args()

    questions = (json.loads(Path(args.questions).read_text(encoding="utf-8"))
                 if args.questions else DEFAULT_QUESTIONS)
    chunks, matrix = store.load(args.store)
    print(f"Loaded {len(chunks)} chunks from {args.store}")

    # One batched embedding call for all questions.
    vecs = embeddings.embed([q["question"] for q in questions])

    scored = []
    for q, vec in zip(questions, vecs):
        hits = store.search(vec, matrix, top_k=args.top_k)
        top_score = hits[0][1] if hits else 0.0
        scored.append((q["label"], top_score, q["question"]))

    print("\nTop retrieval score per question:\n")
    for label, score, question in sorted(scored, key=lambda r: r[1], reverse=True):
        tag = "A" if label == "answerable" else "U"
        print(f"  [{tag}] {score:.3f}  {question}")

    threshold, separable, detail = suggest_threshold(
        [(l, s) for l, s, _ in scored])
    print()
    if separable:
        print(f"Suggested --min-score: {threshold:.3f}")
        print(f"  ({detail})")
        print("  Verify it in chat: answerable questions should answer, "
              "unanswerable ones should refuse.")
    else:
        print("No clean threshold on this store.")
        print(f"  ({detail})")
        print("  This is a finding, not a failure: with very few chunks, "
              "scores compress and the threshold can't separate the classes. "
              "Try a store with more chunks (smaller --chunk-tokens), or rely "
              "on the prompt-level refusal instead.")


if __name__ == "__main__":
    main()
