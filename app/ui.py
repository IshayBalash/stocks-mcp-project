
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import asyncio
import streamlit as st
from langchain_core.messages import HumanMessage, AIMessage

from app.agent import init_agent, astream_response

st.set_page_config(page_title="Stock Assistant", page_icon="📈")
st.title("📈 Stock Assistant")

# ── Persistent event loop ─────────────────────────────────────────────────────
#
# asyncio.run() creates a NEW event loop each time it's called.
# The MCP SSE connections created inside init_agent() are tied to that loop.
# If we later call astream_response() in a different loop (e.g. in a thread),
# those connections break — tools silently fail.
#
# Fix: create ONE event loop at startup and reuse it for every async call.
#
if "loop" not in st.session_state:
    st.session_state.loop = asyncio.new_event_loop()

def run_async(coro):
    """Run an async coroutine on the persistent session event loop."""
    return st.session_state.loop.run_until_complete(coro)


# ── Initialize agent once ─────────────────────────────────────────────────────
if "agent" not in st.session_state:
    with st.spinner("Connecting to MCP server..."):
        st.session_state.agent = run_async(init_agent())

if "messages" not in st.session_state:
    st.session_state.messages = []  # [{"role": "user"/"assistant", "content": str}]

# ── Render chat history ───────────────────────────────────────────────────────
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])


# ── Chat input ────────────────────────────────────────────────────────────────
if prompt := st.chat_input("Ask about your portfolio or any stock..."):

    # Show user message
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Build LangChain message list from full history
    lc_messages = []
    for msg in st.session_state.messages:
        if msg["role"] == "user":
            lc_messages.append(HumanMessage(msg["content"]))
        else:
            lc_messages.append(AIMessage(msg["content"]))

    # Stream assistant response
    with st.chat_message("assistant"):
        tool_container = st.container()
        text_placeholder = st.empty()
        response_text = ""

        # Collect response event by event on the persistent loop.
        # We can't iterate an async generator synchronously directly,
        # so we pull one item at a time using __anext__().
        gen = astream_response(st.session_state.agent, lc_messages)
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
