"""
RAG - Step 1: What is an embedding?

An embedding is a list of numbers that represents the *meaning* of a piece of text.
Texts with similar meaning produce similar vectors.

We'll embed 5 sentences and measure how close they are to each other.
"""

from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

# Load a small, fast model (downloads ~90MB on first run)
model = SentenceTransformer("all-MiniLM-L6-v2")

sentences = [
    "I prefer low risk investments",           # A - user profile
    "I don't like volatile stocks",            # B - similar meaning to A
    "I'm interested in Israeli tech stocks",   # C - different topic
    "What is the weather in Tel Aviv today?",  # D - completely unrelated
    "I want safe, stable, long-term growth",   # E - similar meaning to A
]

# Each sentence becomes a vector of 384 numbers
embeddings = model.encode(sentences)

print(f"Each embedding is a vector of {len(embeddings[0])} numbers\n")
print(f"First embedding (truncated): {embeddings[0][:8].round(3)} ...\n")

# Cosine similarity: 1.0 = identical meaning, 0.0 = unrelated, -1.0 = opposite
similarity_matrix = cosine_similarity(embeddings)

print("Similarity scores (1.0 = identical meaning):\n")
for i in range(len(sentences)):
    for j in range(i + 1, len(sentences)):
        score = similarity_matrix[i][j]
        print(f"  [{score:.2f}]  \"{sentences[i]}\"")
        print(f"         \"{sentences[j]}\"")
        print()
