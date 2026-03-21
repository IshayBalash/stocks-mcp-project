"""
UserStore — manages the users table in Postgres.

Table: users
  user_id    TEXT        — UUID, primary key
  name       TEXT        — display name
  created_at TIMESTAMPTZ — when the user was created
"""

import os
import uuid
from typing import Optional

import psycopg2
from dotenv import load_dotenv

load_dotenv()


class UserStore:

    def __init__(self):
        self._url = os.getenv("POSTGRES_URL")

    def _connect(self):
        return psycopg2.connect(self._url)

    # ── Init ──────────────────────────────────────────────────────────────────

    def init_db(self):
        """Create the users table if it doesn't exist. Call once at startup."""
        with self._connect() as conn:
            with conn.cursor() as cur:
                cur.execute("""
                    CREATE TABLE IF NOT EXISTS users (
                        user_id    TEXT NOT NULL PRIMARY KEY,
                        name       TEXT NOT NULL UNIQUE,
                        created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
                    )
                """)
            conn.commit()

    # ── Query ─────────────────────────────────────────────────────────────────

    def list_users(self) -> list[dict]:
        """Return all users, newest first."""
        with self._connect() as conn:
            with conn.cursor() as cur:
                cur.execute("SELECT user_id, name, created_at FROM users ORDER BY created_at DESC")
                rows = cur.fetchall()
        return [{"user_id": r[0], "name": r[1], "created_at": r[2]} for r in rows]

    def name_exists(self, name: str) -> bool:
        """Return True if a user with this name already exists."""
        with self._connect() as conn:
            with conn.cursor() as cur:
                cur.execute("SELECT 1 FROM users WHERE name = %s", (name,))
                return cur.fetchone() is not None

    def get_user(self, user_id: str) -> Optional[dict]:
        """Return a single user by ID, or None if not found."""
        with self._connect() as conn:
            with conn.cursor() as cur:
                cur.execute("SELECT user_id, name, created_at FROM users WHERE user_id = %s", (user_id,))
                row = cur.fetchone()
        return {"user_id": row[0], "name": row[1], "created_at": row[2]} if row else None

    # ── Write ─────────────────────────────────────────────────────────────────

    def create_user(self, name: str) -> str:
        """Create a new user with a generated UUID. Returns the new user_id."""
        user_id = str(uuid.uuid4())
        with self._connect() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    "INSERT INTO users (user_id, name) VALUES (%s, %s)",
                    (user_id, name)
                )
            conn.commit()
        return user_id


# ── Singleton ──────────────────────────────────────────────────────────────────
user_store = UserStore()
