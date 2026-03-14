"""
RAG - Step 4: Proper profile vector store

Table design:
  - user_id:    which user this belongs to
  - category:   topic label (risk_tolerance, markets, experience, horizon, style)
  - text:       human-readable description of the user's profile in this category
  - embedding:  vector representation of the text
  - updated_at: when this chunk was last updated

One row per (user_id, category) — upsert replaces the chunk when updated.
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

conn = psycopg2.connect(os.getenv("POSTGRES_URL"))
register_vector(conn)
cur = conn.cursor()

# ── Create table ──────────────────────────────────────────────────────────────
cur.execute("""
    CREATE TABLE IF NOT EXISTS user_profile (
        user_id     TEXT NOT NULL,
        category    TEXT NOT NULL,
        text        TEXT NOT NULL,
        embedding   vector(384) NOT NULL,
        updated_at  TIMESTAMPTZ NOT NULL DEFAULT NOW(),
        PRIMARY KEY (user_id, category)
    )
""")
conn.commit()


# ── Helpers ───────────────────────────────────────────────────────────────────
def upsert_chunk(user_id: str, category: str, text: str):
    """Insert or replace a profile chunk for a user."""
    embedding = model.encode([text])[0].tolist()
    cur.execute("""
        INSERT INTO user_profile (user_id, category, text, embedding)
        VALUES (%s, %s, %s, %s)
        ON CONFLICT (user_id, category) DO UPDATE
            SET text       = EXCLUDED.text,
                embedding  = EXCLUDED.embedding,
                updated_at = NOW()
    """, (user_id, category, text, embedding))
    conn.commit()
    print(f"  upserted [{category}]: {text}")


def retrieve(user_id: str, query: str, top_k: int = 2, min_score: float = 0.3) -> list[dict]:
    """Find the most relevant profile chunks for this user and query."""
    query_embedding = model.encode([query])[0].tolist()
    cur.execute("""
        SELECT category, text, 1 - (embedding <=> %s::vector) AS similarity
        FROM user_profile
        WHERE user_id = %s
        ORDER BY similarity DESC
        LIMIT %s
    """, (query_embedding, user_id, top_k))
    results = [{"category": r[0], "text": r[1], "score": r[2]} for r in cur.fetchall()]
    return [r for r in results if r["score"] >= min_score]


# ── Test ──────────────────────────────────────────────────────────────────────
USER = "user_1"

print("=== Initial profile ===")
upsert_chunk(USER, "risk_tolerance", "User prefers low risk investments and avoids volatile stocks.")
upsert_chunk(USER, "markets",        "User is focused on US markets only.")
upsert_chunk(USER, "experience",     "User has 3 years of investing experience.")
upsert_chunk(USER, "horizon",        "User has a long-term investment horizon of 10+ years.")

print("\n=== Query: risk ===")
for r in retrieve(USER, "What is the user's risk tolerance?"):
    print(f"  [{r['score']:.2f}] ({r['category']}) {r['text']}")

print("\n=== User explicitly updates markets preference ===")
upsert_chunk(USER, "markets", "User is interested in both US markets and Israeli tech stocks on TASE.")

print("\n=== Query: markets (after update) ===")
for r in retrieve(USER, "Is the user interested in any specific markets?"):
    print(f"  [{r['score']:.2f}] ({r['category']}) {r['text']}")

cur.close()
conn.close()
