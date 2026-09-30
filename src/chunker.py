"""Split text into overlapping token windows.

Why chunk at all? Two reasons:
1. Embedding models have input limits, and one giant vector for a whole
   document is a blurry summary — retrieval needs focused units.
2. The LLM's context window is finite — we can only afford to send the
   most relevant chunks, not the whole document.

Overlap exists so a sentence split across a boundary still appears whole
in at least one chunk.
"""
import tiktoken

ENCODING = "o200k_base"


def chunk_text(text: str, chunk_tokens: int = 500,
               overlap_tokens: int = 50) -> list[str]:
    enc = tiktoken.get_encoding(ENCODING)
    tokens = enc.encode(text)
    if not tokens:
        return []
    chunks = []
    step = chunk_tokens - overlap_tokens
    start = 0
    while start < len(tokens):
        end = start + chunk_tokens
        chunks.append(enc.decode(tokens[start:end]))
        if end >= len(tokens):
            break
        start += step
    return chunks


# ---- EXERCISES ----
# 1. Try chunk_tokens=200 vs 1000 on your statement. Which retrieves better?
#    (Hint: point your prompt-lab harness at this pipeline to measure it.)
# 2. This splitter can cut mid-sentence. Improve it: split on paragraph
#    boundaries first, then pack paragraphs into token windows.
# 3. Add chunk metadata: page numbers. Requires tracking token->page offsets
#    in pdf_reader — how would you do it?
