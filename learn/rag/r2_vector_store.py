"""
RAG - Step 2: A simple in-memory vector store

This simulates what pgvector does:
  - Store documents with their embeddings
  - Given a query, find the most relevant documents

No DB yet — pure Python. Same logic, just in RAM.
"""

from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

model = SentenceTransformer("all-MiniLM-L6-v2")

# ── The "documents" we store (imagine these are chunks of the user profile) ──
documents = [
    "User prefers low risk investments and avoids volatile stocks.",
    "User is interested in Israeli tech stocks listed on TASE.",
    "User has a long-term investment horizon of 10+ years.",
    "User avoids leverage and prefers stable dividend-paying companies.",
    "User has 3 years of investing experience, mostly in US markets.",
]

# Embed all documents once (this is the "ingestion" step)
doc_embeddings = model.encode(documents)
print(f"Stored {len(documents)} documents, each as a {len(doc_embeddings[0])}-dim vector\n")


def retrieve(query: str, top_k: int = 2, min_score: float = 0.3) -> list[dict]:
    """Find the most relevant documents for a given query."""
    query_embedding = model.encode([query])
    scores = cosine_similarity(query_embedding, doc_embeddings)[0]

    # Pair each document with its score, sort by score descending
    results = sorted(
        [{"text": doc, "score": float(score)} for doc, score in zip(documents, scores)],
        key=lambda x: x["score"],
        reverse=True,
    )

    # Apply top_k and min_score threshold
    results = [r for r in results[:top_k] if r["score"] >= min_score]
    return results


# ── Test queries ──────────────────────────────────────────────────────────────
queries = [
    "What is the user's risk tolerance?",
    "Is the user interested in any specific markets?",
    "How experienced is this investor?",
    "What is the user's favorite food?",   # should return nothing
]

for query in queries:
    print(f"Query: \"{query}\"")
    results = retrieve(query)
    if results:
        for r in results:
            print(f"  [{r['score']:.2f}] {r['text']}")
    else:
        print("  (no relevant documents found)")
    print()
