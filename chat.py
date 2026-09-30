#!/usr/bin/env python3
"""Chat with an ingested document: load the store, then a REPL question loop.

Usage:
    python ingest.py --pdf statement.pdf --store ./store   # one time
    python chat.py --store ./store
"""
import argparse

from src import rag, store


def main():
    parser = argparse.ArgumentParser(description="Chat with your document.")
    parser.add_argument("--store", required=True, help="Vector store directory")
    parser.add_argument("--top-k", type=int, default=3,
                        help="How many chunks to retrieve")
    parser.add_argument("--min-score", type=float, default=0.25,
                        help="Refuse below this similarity score")
    args = parser.parse_args()

    chunks, matrix = store.load(args.store)
    print(f"Loaded {len(chunks)} chunks. Ask questions (empty line quits).\n")

    total_cost = 0.0
    while True:
        try:
            question = input("you> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if not question:
            break
        result = rag.answer(question, chunks, matrix,
                            top_k=args.top_k, min_score=args.min_score)
        total_cost += result["cost_usd"]
        print(f"\ndoc> {result['answer']}")
        if not result["refused"]:
            scores = ", ".join(f"{s:.2f}" for _, s in result["hits"])
            print(f"    [retrieved chunk scores: {scores} | "
                  f"~${result['cost_usd']:.4f}]")
        print()
    print(f"Session cost: ~${total_cost:.4f}")


if __name__ == "__main__":
    main()
