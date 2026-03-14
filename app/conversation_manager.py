"""
ConversationManager — combines ConversationCache and ConversationStore into one interface.

This is the only class the rest of the app should import and use.
It handles the two-layer cache logic:
  - load: Redis first, fall back to Postgres
  - save: write to both
  - delete: remove from both
"""

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage

from app.conversation_cache import ConversationCache
from app.conversation_store import ConversationStore


class ConversationManager:

    def __init__(self):
        self.redis = ConversationCache()
        self.postgres = ConversationStore()

    # ── Init ──────────────────────────────────────────────────────────────────

    def init_db(self):
        """Create Postgres tables if they don't exist. Call once at startup."""
        self.postgres.init_db()

    # ── Query ─────────────────────────────────────────────────────────────────

    def load_history(self, user_id: str, conversation_id: str) -> list[BaseMessage]:
        """
        Load messages for a conversation as LangChain message objects.
        Redis hit  → return immediately (fast).
        Redis miss → load from Postgres, warm Redis, return.
        """
        messages = self.redis.get_messages(user_id, conversation_id)

        if not messages:
            messages = self.postgres.load_messages(user_id, conversation_id)
            if messages:
                self.redis.warm(user_id, conversation_id, messages, created_at=self._now())

        return [self._dict_to_msg(m) for m in messages]

    def list_conversations(self, user_id: str) -> list[dict]:
        """Return all conversations for a user from Postgres (newest first)."""
        return self.postgres.list_conversations(user_id)

    # ── Insert ────────────────────────────────────────────────────────────────

    def save_turn(self, user_id: str, conversation_id: str, user_msg: str, assistant_msg: str):
        """
        Append one turn and persist to both Redis and Postgres.
        Called after streaming completes.
        """
        # Redis
        self.redis.append_turn(user_id, conversation_id, user_msg, assistant_msg)

        # Postgres — append directly in SQL, no RAM load needed
        self.postgres.append_turn(user_id, conversation_id, user_msg, assistant_msg)

    # ── Delete ────────────────────────────────────────────────────────────────

    def delete_conversation(self, user_id: str, conversation_id: str):
        """Delete a conversation from both Redis and Postgres."""
        self.redis.delete(user_id, conversation_id)
        self.postgres.delete_conversation(user_id, conversation_id)

    def delete_all_conversations(self, user_id: str):
        """Delete all conversations for a user from both stores."""
        conversations = self.postgres.list_conversations(user_id)
        for c in conversations:
            self.redis.delete(user_id, c["conversation_id"])
        self.postgres.delete_all_conversations(user_id)

    # ── Helpers ───────────────────────────────────────────────────────────────

    def _dict_to_msg(self, d: dict) -> BaseMessage:
        return HumanMessage(d["content"]) if d["role"] == "user" else AIMessage(d["content"])

    def _now(self) -> str:
        from datetime import datetime, timezone
        return datetime.now(timezone.utc).isoformat()


# ── Singleton ─────────────────────────────────────────────────────────────────
# Import this instance everywhere — don't instantiate DBManager directly.

db = ConversationManager()
