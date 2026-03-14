"""
ConversationStore — handles all PostgreSQL operations for chat history.

Table schema:
  chat_history(
    user_id         TEXT,
    conversation_id TEXT,
    created_at      TIMESTAMPTZ DEFAULT NOW(),
    messages        JSONB DEFAULT '[]',
    PRIMARY KEY (user_id, conversation_id)
  )
"""

import json
import os

import psycopg2
from dotenv import load_dotenv

load_dotenv()


class ConversationStore:

    def __init__(self):
        self._url = os.getenv("POSTGRES_URL")

    def _connect(self):
        return psycopg2.connect(self._url)

    # ── DB init ───────────────────────────────────────────────────────────────

    def init_db(self):
        """Create the chat_history table if it doesn't exist. Call once at startup."""
        with self._connect() as conn:
            with conn.cursor() as cur:
                cur.execute("""
                    CREATE TABLE IF NOT EXISTS chat_history (
                        user_id         TEXT NOT NULL,
                        conversation_id TEXT NOT NULL,
                        created_at      TIMESTAMPTZ NOT NULL DEFAULT NOW(),
                        messages        JSONB NOT NULL DEFAULT '[]',
                        PRIMARY KEY (user_id, conversation_id)
                    )
                """)
            conn.commit()

    # ── Query ─────────────────────────────────────────────────────────────────

    def load_messages(self, user_id: str, conversation_id: str) -> list[dict]:
        """Return the full messages list for a conversation, or [] if not found."""
        with self._connect() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    "SELECT messages FROM chat_history WHERE user_id = %s AND conversation_id = %s",
                    (user_id, conversation_id)
                )
                row = cur.fetchone()
        return row[0] if row else []

    def list_conversations(self, user_id: str) -> list[dict]:
        """Return all conversations for a user, newest first."""
        with self._connect() as conn:
            with conn.cursor() as cur:
                cur.execute("""
                    SELECT conversation_id, created_at, jsonb_array_length(messages)
                    FROM chat_history
                    WHERE user_id = %s
                    ORDER BY created_at DESC
                """, (user_id,))
                rows = cur.fetchall()
        return [
            {"conversation_id": r[0], "created_at": r[1], "message_count": r[2]}
            for r in rows
        ]

    # ── Insert / Update ───────────────────────────────────────────────────────

    def append_turn(self, user_id: str, conversation_id: str, user_msg: str, assistant_msg: str):
        """
        Append one turn directly in SQL — no RAM load needed.
        INSERT on first turn, || (JSONB concat) on conflict.
        """
        new_messages = json.dumps([
            {"role": "user", "content": user_msg},
            {"role": "assistant", "content": assistant_msg},
        ])
        with self._connect() as conn:
            with conn.cursor() as cur:
                cur.execute("""
                    INSERT INTO chat_history (user_id, conversation_id, messages)
                    VALUES (%s, %s, %s::jsonb)
                    ON CONFLICT (user_id, conversation_id) DO UPDATE
                        SET messages = chat_history.messages || EXCLUDED.messages
                """, (user_id, conversation_id, new_messages))
            conn.commit()

    def save_messages(self, user_id: str, conversation_id: str, messages: list[dict]):
        """
        Upsert messages for a conversation.
        INSERT on first turn (created_at set by DB default).
        UPDATE on subsequent turns (created_at never changes).
        """
        with self._connect() as conn:
            with conn.cursor() as cur:
                cur.execute("""
                    INSERT INTO chat_history (user_id, conversation_id, messages)
                    VALUES (%s, %s, %s)
                    ON CONFLICT (user_id, conversation_id) DO UPDATE
                        SET messages = EXCLUDED.messages
                """, (user_id, conversation_id, json.dumps(messages)))
            conn.commit()

    # ── Delete ────────────────────────────────────────────────────────────────

    def delete_conversation(self, user_id: str, conversation_id: str):
        """Delete a single conversation."""
        with self._connect() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    "DELETE FROM chat_history WHERE user_id = %s AND conversation_id = %s",
                    (user_id, conversation_id)
                )
            conn.commit()

    def delete_all_conversations(self, user_id: str):
        """Delete all conversations for a user."""
        with self._connect() as conn:
            with conn.cursor() as cur:
                cur.execute("DELETE FROM chat_history WHERE user_id = %s", (user_id,))
            conn.commit()
