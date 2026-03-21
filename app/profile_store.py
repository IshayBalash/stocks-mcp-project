"""
ProfileStore — stores and retrieves user profile chunks using pgvector.

Each chunk represents one aspect of the user's investment profile (e.g. risk tolerance,
markets, experience). One row per (user_id, category) — upsert replaces on update.

Table: user_profile
  user_id     TEXT        — which user
  category    TEXT        — topic label (risk_tolerance, markets, experience, horizon, style)
  text        TEXT        — human-readable profile description for this category
  embedding   vector(384) — semantic embedding of the text
  updated_at  TIMESTAMPTZ — when this chunk was last written
"""

import contextlib
import io
import json
import os
from typing import Optional

import psycopg2
from dotenv import load_dotenv
from pgvector.psycopg2 import register_vector
from sentence_transformers import SentenceTransformer

load_dotenv()

EMBEDDING_MODEL = "all-MiniLM-L6-v2"
EMBEDDING_DIM = 384
MIN_SCORE = 0.3

VALID_CATEGORIES = {"risk_tolerance", "markets", "experience", "horizon", "style"}


class ProfileStore:

    def __init__(self):
        self._url = os.getenv("POSTGRES_URL")
        self._model: Optional[SentenceTransformer] = None

    @property
    def model(self) -> SentenceTransformer:
        """Lazy load — model only loaded into RAM when first used."""
        if self._model is None:
            with contextlib.redirect_stdout(io.StringIO()):
                self._model = SentenceTransformer(EMBEDDING_MODEL)
        return self._model

    def _connect(self):
        conn = psycopg2.connect(self._url)
        register_vector(conn)
        return conn

    # ── Init ──────────────────────────────────────────────────────────────────

    def init_db(self):
        """Create the user_profile table if it doesn't exist. Call once at startup."""
        with self._connect() as conn:
            with conn.cursor() as cur:
                cur.execute("CREATE EXTENSION IF NOT EXISTS vector")
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

    # ── Write ─────────────────────────────────────────────────────────────────

    def upsert(self, user_id: str, category: str, text: str):
        """
        Insert or replace a profile chunk for a user.
        If the category already exists, the text and embedding are overwritten.
        """
        if category not in VALID_CATEGORIES:
            raise ValueError(f"Invalid category '{category}'. Must be one of: {VALID_CATEGORIES}")

        embedding = self.model.encode([text])[0].tolist()
        with self._connect() as conn:
            with conn.cursor() as cur:
                cur.execute("""
                    INSERT INTO user_profile (user_id, category, text, embedding)
                    VALUES (%s, %s, %s, %s)
                    ON CONFLICT (user_id, category) DO UPDATE
                        SET text       = EXCLUDED.text,
                            embedding  = EXCLUDED.embedding,
                            updated_at = NOW()
                """, (user_id, category, text, embedding))
            conn.commit()

    # ── Read ──────────────────────────────────────────────────────────────────

    def retrieve(self, user_id: str, query: str, top_k: int = 3) -> list[dict]:
        """
        Find the most relevant profile chunks for a given query.
        Returns a list of {category, text, score} dicts, filtered by MIN_SCORE.
        Returns [] if no relevant chunks are found.
        """
        query_embedding = self.model.encode([query])[0].tolist()
        with self._connect() as conn:
            with conn.cursor() as cur:
                cur.execute("""
                    SELECT category, text, 1 - (embedding <=> %s::vector) AS score
                    FROM user_profile
                    WHERE user_id = %s
                    ORDER BY score DESC
                    LIMIT %s
                """, (query_embedding, user_id, top_k))
                rows = cur.fetchall()

        results = [{"category": r[0], "text": r[1], "score": r[2]} for r in rows]
        return [r for r in results if r["score"] >= MIN_SCORE]

    def get_all(self, user_id: str) -> list[dict]:
        """Return all profile chunks for a user (for display in UI)."""
        with self._connect() as conn:
            with conn.cursor() as cur:
                cur.execute("""
                    SELECT category, text, updated_at
                    FROM user_profile
                    WHERE user_id = %s
                    ORDER BY category
                """, (user_id,))
                rows = cur.fetchall()
        return [{"category": r[0], "text": r[1], "updated_at": r[2]} for r in rows]

    def format_for_prompt(self, chunks: list[dict]) -> str:
        """
        Format retrieved chunks into a string to inject into the system prompt.
        Returns empty string if no chunks.
        """
        if not chunks:
            return ""
        lines = ["User profile context:"]
        for c in chunks:
            lines.append(f"  [{c['category']}] {c['text']}")
        return "\n".join(lines)


# ── Singleton ──────────────────────────────────────────────────────────────────
profile_store = ProfileStore()
