import datetime

import streamlit as st

from agent import stream_agent_response

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(page_title="Stock Assistant", page_icon="📈", layout="centered")
st.title("📈 Stock Assistant")
st.caption("Ask me about stocks, your portfolio, or market data.")

# ── Constants ─────────────────────────────────────────────────────────────────
CONTEXT_WINDOW = 20  # max non-system messages sent to the LLM per turn

# ── Session state init ────────────────────────────────────────────────────────
if "messages" not in st.session_state:
    st.session_state.messages = [
        {
            "role": "system",
            "content": (
                f"You are a helpful stock market assistant. Today's date is {datetime.date.today()}. "
                "You have access to tools that can fetch stock prices, read the user's portfolio, "
                "and analyze trading data. Use tools when needed and provide clear, concise answers."
            ),
        }
    ]

# ── Helpers ───────────────────────────────────────────────────────────────────
def build_context(messages: list) -> list:
    """
    Return the system message + the last CONTEXT_WINDOW non-system messages.
    This caps how much history is sent to the LLM, keeping token usage bounded.
    The full history remains in st.session_state for display.
    """
    system = [m for m in messages if m["role"] == "system"]
    non_system = [m for m in messages if m["role"] != "system"]
    return system + non_system[-CONTEXT_WINDOW:]


def is_displayable(msg: dict) -> bool:
    """
    Only render user and assistant messages that have visible text content.
    Skips: system messages, tool results, and assistant messages that only
    contain tool_calls with no text (internal plumbing, not user-facing).
    """
    if msg["role"] not in ("user", "assistant"):
        return False
    return bool(msg.get("content"))


# ── Display full chat history ─────────────────────────────────────────────────
for msg in st.session_state.messages:
    if is_displayable(msg):
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

# ── Chat input ────────────────────────────────────────────────────────────────
if prompt := st.chat_input("Ask about a stock..."):

    # 1. Save and display the user message
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # 2. Build windowed context and snapshot its length before the agent runs.
    #    The agent will append intermediate messages (tool calls, tool results)
    #    to this list during its run.
    context = build_context(st.session_state.messages)
    context_len_before = len(context)

    # 3. Stream the assistant response
    with st.chat_message("assistant"):
        full_response = st.write_stream(stream_agent_response(context))

    # 4. Sync any intermediate messages the agent added (tool calls + tool
    #    results) back into the full session state so future context builds
    #    include them.
    new_intermediate = context[context_len_before:]
    st.session_state.messages.extend(new_intermediate)

    # 5. Append the final assistant text message
    st.session_state.messages.append({"role": "assistant", "content": full_response})