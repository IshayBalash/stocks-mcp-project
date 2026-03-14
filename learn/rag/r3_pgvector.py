"""
RAG - Step 3: Store and retrieve embeddings in pgvector (Postgres)

Same logic as r2, but now embeddings are persisted in Postgres.
Survives restarts. Queryable from anywhere.

pgvector operators:
  <->  L2 distance       (lower = more similar)
  <#>  inner product     (higher = more similar)
  <=>  cosine distance   (lower = more similar, range 0-2, 0 = identical)
"""

import os
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from dotenv import load_dotenv
load_dotenv()

import psycopg2
from pgvector.psycopg2 import register_vector
from sentence_transformers import SentenceTransformer

model = SentenceTransformer("all-MiniLM-L6-v2")

# ── Connect to Postgres ───────────────────────────────────────────────────────
conn = psycopg2.connect(os.getenv("POSTGRES_URL"))
register_vector(conn)  # teaches psycopg2 how to read/write vector columns
cur = conn.cursor()

# ── Create table ──────────────────────────────────────────────────────────────
# vector(384) means each row stores a 384-dimensional vector
cur.execute("""
    CREATE TABLE IF NOT EXISTS learn_profile_chunks (
        id      SERIAL PRIMARY KEY,
        text    TEXT NOT NULL,
        embedding vector(384) NOT NULL
    )
""")
conn.commit()

# ── Insert documents (only if table is empty) ─────────────────────────────────
cur.execute("SELECT COUNT(*) FROM learn_profile_chunks")
if cur.fetchone()[0] == 0:
    documents = [
        "User prefers low risk investments and avoids volatile stocks.",
        "User is interested in Israeli tech stocks listed on TASE.",
        "User has a long-term investment horizon of 10+ years.",
        "User avoids leverage and prefers stable dividend-paying companies.",
        "User has 3 years of investing experience, mostly in US markets.",
    ]
    embeddings = model.encode(documents)
    for text, embedding in zip(documents, embeddings):
        cur.execute(
            "INSERT INTO learn_profile_chunks (text, embedding) VALUES (%s, %s)",
            (text, embedding.tolist())
        )
    conn.commit()
    print(f"Inserted {len(documents)} documents\n")
else:
    print("Documents already in DB, skipping insert\n")


# ── Retrieve ──────────────────────────────────────────────────────────────────
def retrieve(query: str, top_k: int = 2, min_score: float = 0.3) -> list[dict]:
    query_embedding = model.encode([query])[0]

    # <=> is cosine distance: 0 = identical, 2 = opposite
    # We convert to similarity: similarity = 1 - distance
    cur.execute("""
        SELECT text, 1 - (embedding <=> %s::vector) AS similarity
        FROM learn_profile_chunks
        ORDER BY similarity DESC
        LIMIT %s
    """, (query_embedding.tolist(), top_k))

    results = [{"text": row[0], "score": row[1]} for row in cur.fetchall()]
    return [r for r in results if r["score"] >= min_score]


# ── Test queries ──────────────────────────────────────────────────────────────
queries = [
    "What is the user's risk tolerance?",
    "Is the user interested in any specific markets?",
    "How experienced is this investor?",
    "What is the user's favorite food?",
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

cur.close()
conn.close()
