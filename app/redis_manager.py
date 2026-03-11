"""
RedisManager — handles all Redis operations for chat history.

Each conversation is stored as one Redis key:
  Key:   chat_history:{user_id}:{conversation_id}
  Value: {"created_at": "2026-03-10T10:00:00+00:00", "messages": [...]}

Messages are capped at MAX_MESSAGES (ring buffer).
Keys expire after REDIS_TTL seconds.
"""

import json
import os
from datetime import datetime, timezone
from typing import Optional

import redis
from dotenv import load_dotenv

load_dotenv()

MAX_MESSAGES = 20
REDIS_TTL = 60 * 60 * 24  # 24 h


class RedisManager:

    def __init__(self):
        self._client: Optional[redis.Redis] = None
        self._url = os.getenv("REDIS_URL", "redis://localhost:6379")

    @property
    def client(self) -> redis.Redis:
        """Lazy connection — created on first use."""
        if self._client is None:
            self._client = redis.from_url(self._url, decode_responses=True)
        return self._client

    # ── Key helpers ───────────────────────────────────────────────────────────

    def _key(self, user_id: str, conversation_id: str) -> str:
        return f"chat_history:{user_id}:{conversation_id}"

    def _now(self) -> str:
        return datetime.now(timezone.utc).isoformat()

    # ── Query ─────────────────────────────────────────────────────────────────

    def get(self, user_id: str, conversation_id: str) -> Optional[dict]:
        """
        Return the full stored dict {"created_at": ..., "messages": [...]}
        or None if the key doesn't exist.
        """
        raw = self.client.get(self._key(user_id, conversation_id))
        return json.loads(raw) if raw else None

    def get_messages(self, user_id: str, conversation_id: str) -> list[dict]:
        """Return just the messages list, or [] if not found."""
        data = self.get(user_id, conversation_id)
        return data["messages"] if data else []

    # ── Insert / Update ───────────────────────────────────────────────────────

    def append_turn(self, user_id: str, conversation_id: str, user_msg: str, assistant_msg: str):
        """
        Append one turn (user + assistant) to the conversation.
        Creates the key with a created_at timestamp if it doesn't exist yet.
        """
        data = self.get(user_id, conversation_id)

        if data:
            created_at = data["created_at"]
            messages = data["messages"]
        else:
            created_at = self._now()
            messages = []

        messages.append({"role": "user", "content": user_msg})
        messages.append({"role": "assistant", "content": assistant_msg})
        messages = messages[-MAX_MESSAGES:]  # keep ring buffer capped

        self.client.set(
            self._key(user_id, conversation_id),
            json.dumps({"created_at": created_at, "messages": messages}),
            ex=REDIS_TTL,
        )

    def warm(self, user_id: str, conversation_id: str, messages: list[dict], created_at: str):
        """
        Write messages into Redis (called after a cold load from Postgres).
        Only stores the last MAX_MESSAGES to keep the ring buffer capped.
        """
        self.client.set(
            self._key(user_id, conversation_id),
            json.dumps({"created_at": created_at, "messages": messages[-MAX_MESSAGES:]}),
            ex=REDIS_TTL,
        )

    # ── Delete ────────────────────────────────────────────────────────────────

    def delete(self, user_id: str, conversation_id: str):
        """Delete a single conversation from Redis."""
        self.client.delete(self._key(user_id, conversation_id))
