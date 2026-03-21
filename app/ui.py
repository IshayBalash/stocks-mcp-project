
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import asyncio
import uuid
import streamlit as st

from app.agent import init_agent, astream_response
from app.conversation_manager import db
from app.user_store import user_store
from app.profile_store import profile_store

st.set_page_config(page_title="Stock Assistant", page_icon="📈", layout="wide")

# ── Persistent event loop ─────────────────────────────────────────────────────
if "loop" not in st.session_state:
    st.session_state.loop = asyncio.new_event_loop()

def run_async(coro):
    return st.session_state.loop.run_until_complete(coro)


# ── Initialize agent once ─────────────────────────────────────────────────────
if "agent" not in st.session_state:
    with st.spinner("Connecting to MCP server..."):
        st.session_state.agent = run_async(init_agent())


# ── Page: User picker ─────────────────────────────────────────────────────────
def show_user_picker():
    st.title("📈 Stock Assistant")
    st.subheader("Who are you?")

    users = user_store.list_users()

    if users:
        st.write("Select your account:")
        cols = st.columns(min(len(users), 4))
        for i, user in enumerate(users):
            with cols[i % 4]:
                if st.button(f"👤 {user['name']}", key=user["user_id"], use_container_width=True):
                    st.session_state.user_id = user["user_id"]
                    st.session_state.user_name = user["name"]
                    st.session_state.conversation_id = str(uuid.uuid4())
                    st.session_state.messages = []
                    st.rerun()
        st.divider()

    if st.button("＋ New user", type="primary"):
        st.session_state.show_registration = True
        st.rerun()


# ── Page: Registration ────────────────────────────────────────────────────────
def show_registration():
    st.title("📈 Stock Assistant")
    st.subheader("Create your account")

    with st.form("registration_form"):
        name = st.text_input("Your name")

        st.markdown("**Investment profile**")

        risk = st.radio(
            "Risk tolerance",
            ["Low — I prefer stable, low-risk investments",
             "Medium — I accept some volatility for better returns",
             "High — I'm comfortable with high risk for high reward"],
        )
        markets = st.multiselect(
            "Markets you invest in",
            ["US (NYSE / NASDAQ)", "Israel (TASE)", "Europe", "Other"],
            default=["US (NYSE / NASDAQ)"],
        )
        experience = st.radio(
            "Investing experience",
            ["Less than 1 year", "1–3 years", "3–10 years", "10+ years"],
        )
        horizon = st.radio(
            "Investment horizon",
            ["Short term (< 1 year)", "Medium term (1–5 years)", "Long term (5+ years)"],
        )
        style = st.multiselect(
            "Investment style",
            ["Growth", "Value", "Dividend income", "Index / passive", "Mixed"],
            default=["Mixed"],
        )

        submitted = st.form_submit_button("Create account", type="primary")

    if st.button("← Back"):
        st.session_state.show_registration = False
        st.rerun()

    if submitted:
        if not name.strip():
            st.error("Please enter your name.")
            return
        if user_store.name_exists(name.strip()):
            st.error(f"The name '{name.strip()}' is already taken. Please choose a different name.")
            return

        user_id = user_store.create_user(name.strip())

        profile_store.upsert(user_id, "risk_tolerance", f"Risk tolerance: {risk}")
        profile_store.upsert(user_id, "markets",        f"Markets: {', '.join(markets)}")
        profile_store.upsert(user_id, "experience",     f"Investing experience: {experience}")
        profile_store.upsert(user_id, "horizon",        f"Investment horizon: {horizon}")
        profile_store.upsert(user_id, "style",          f"Investment style: {', '.join(style)}")

        st.session_state.user_id = user_id
        st.session_state.user_name = name.strip()
        st.session_state.conversation_id = str(uuid.uuid4())
        st.session_state.messages = []
        st.session_state.show_registration = False
        st.rerun()


# ── Page: Chat ────────────────────────────────────────────────────────────────
def show_chat():
    user_id = st.session_state.user_id

    with st.sidebar:
        st.markdown(f"**{st.session_state.user_name}**")
        if st.button("Switch user", use_container_width=True):
            for key in ["user_id", "user_name", "conversation_id", "messages"]:
                st.session_state.pop(key, None)
            st.rerun()

        st.divider()
        st.header("💬 Conversations")

        if st.button("＋ New conversation", use_container_width=True):
            st.session_state.conversation_id = str(uuid.uuid4())
            st.session_state.messages = []
            st.rerun()

        st.divider()

        conversations = db.list_conversations(user_id)
        if not conversations:
            st.caption("No past conversations yet.")
        else:
            for conv in conversations:
                created = conv["created_at"]
                label = created.strftime("%b %d, %H:%M") if hasattr(created, "strftime") else str(created)[:16]
                count = conv["message_count"]
                is_active = conv["conversation_id"] == st.session_state.conversation_id
                btn_label = f"{'▶ ' if is_active else ''}{label}  ·  {count // 2} turn{'s' if count // 2 != 1 else ''}"
                if st.button(btn_label, key=conv["conversation_id"], use_container_width=True,
                             type="primary" if is_active else "secondary"):
                    if not is_active:
                        st.session_state.conversation_id = conv["conversation_id"]
                        st.session_state.messages = db.postgres.load_messages(user_id, conv["conversation_id"])
                        st.rerun()

    st.title("📈 Stock Assistant")

    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    if prompt := st.chat_input("Ask about your portfolio or any stock..."):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            tool_container = st.container()
            text_placeholder = st.empty()
            response_text = ""

            gen = astream_response(st.session_state.agent, user_id, st.session_state.conversation_id, prompt)
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
        st.rerun()


# ── Router ────────────────────────────────────────────────────────────────────
if "user_id" not in st.session_state:
    if st.session_state.get("show_registration"):
        show_registration()
    else:
        show_user_picker()
else:
    show_chat()
