
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import asyncio
import uuid
import streamlit as st

from app.agent import init_agent, astream_response
from app.conversation_manager import db

USER_ID = "user_1"  # hardcoded for now; will come from auth later

st.set_page_config(page_title="Stock Assistant", page_icon="📈", layout="wide")

# ── Persistent event loop ─────────────────────────────────────────────────────
if "loop" not in st.session_state:
    st.session_state.loop = asyncio.new_event_loop()

def run_async(coro):
    """Run an async coroutine on the persistent session event loop."""
    return st.session_state.loop.run_until_complete(coro)


# ── Initialize agent once ─────────────────────────────────────────────────────
if "conversation_id" not in st.session_state:
    st.session_state.conversation_id = str(uuid.uuid4())

if "agent" not in st.session_state:
    with st.spinner("Connecting to MCP server..."):
        st.session_state.agent = run_async(init_agent())

if "messages" not in st.session_state:
    st.session_state.messages = []


# ── Sidebar: conversation history ─────────────────────────────────────────────
with st.sidebar:
    st.header("💬 Conversations")

    if st.button("＋ New conversation", use_container_width=True):
        st.session_state.conversation_id = str(uuid.uuid4())
        st.session_state.messages = []
        st.rerun()

    st.divider()

    conversations = db.list_conversations(USER_ID)
    if not conversations:
        st.caption("No past conversations yet.")
    else:
        for conv in conversations:
            created = conv["created_at"]
            # created_at is a datetime object from psycopg2
            label = created.strftime("%b %d, %H:%M") if hasattr(created, "strftime") else str(created)[:16]
            count = conv["message_count"]
            is_active = conv["conversation_id"] == st.session_state.conversation_id

            btn_label = f"{'▶ ' if is_active else ''}{label}  ·  {count // 2} turn{'s' if count // 2 != 1 else ''}"
            if st.button(btn_label, key=conv["conversation_id"], use_container_width=True,
                         type="primary" if is_active else "secondary"):
                if not is_active:
                    st.session_state.conversation_id = conv["conversation_id"]
                    st.session_state.messages = db.postgres.load_messages(USER_ID, conv["conversation_id"])
                    st.rerun()


# ── Main chat area ────────────────────────────────────────────────────────────
st.title("📈 Stock Assistant")

# Render chat history
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])


# ── Chat input ────────────────────────────────────────────────────────────────
if prompt := st.chat_input("Ask about your portfolio or any stock..."):

    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        tool_container = st.container()
        text_placeholder = st.empty()
        response_text = ""

        gen = astream_response(st.session_state.agent, USER_ID, st.session_state.conversation_id, prompt)
        while True:
            try:
                event_type, data = run_async(gen.__anext__())
            except StopAsyncIteration:
                break

            if event_type == "tool_start":
                with tool_container:
                    st.caption(f"🔧 Calling `{data}`...")
            elif event_type == "tool_end":
                with tool_container:
                    st.caption(f"✅ `{data}` done")
            elif event_type == "token":
                response_text += data
                text_placeholder.markdown(response_text)

    st.session_state.messages.append({"role": "assistant", "content": response_text})
    st.rerun()  # refresh sidebar so new conversation appears immediately
