#!/usr/bin/env python3
"""Ingest a PDF into a vector store: extract -> chunk -> embed -> save.

Usage:
    python ingest.py --pdf statement.pdf --store ./store
"""
import argparse

from src import chunker, costs, embeddings, pdf_reader, store


def main():
    parser = argparse.ArgumentParser(description="Ingest a PDF into a vector store.")
    parser.add_argument("--pdf", required=True, help="PDF file to ingest")
    parser.add_argument("--store", required=True, help="Directory for the vector store")
    parser.add_argument("--chunk-tokens", type=int, default=500)
    parser.add_argument("--overlap-tokens", type=int, default=50)
    args = parser.parse_args()

    text, pages = pdf_reader.extract_text(args.pdf)
    print(f"Extracted {len(text)} chars from {pages} pages.")

    chunks = chunker.chunk_text(text, args.chunk_tokens, args.overlap_tokens)
    print(f"Split into {len(chunks)} chunks "
          f"({args.chunk_tokens} tokens, {args.overlap_tokens} overlap).")

    total_tokens = sum(costs.count_tokens(c) for c in chunks)
    print(f"Embedding {total_tokens} tokens "
          f"(~${costs.estimate_embed_cost(total_tokens):.4f})...")
    vectors = embeddings.embed(chunks)

    store.save(args.store, chunks, vectors)
    print("Done. Ask questions with: python chat.py --store ./store")


if __name__ == "__main__":
    main()
