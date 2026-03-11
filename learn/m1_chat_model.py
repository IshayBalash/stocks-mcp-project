"""
Module 1.1 — Chat Models & Messages
====================================

WHAT THIS FILE TEACHES:
  - How to instantiate a Chat Model (ChatOpenAI pointed at OpenRouter)
  - The four message types: SystemMessage, HumanMessage, AIMessage, ToolMessage
  - How to call .invoke() and read the response
  - How the full message list represents conversation history

MENTAL MODEL:
  Every LLM call = you hand it a LIST of messages, it returns ONE AIMessage.
  The model has no memory on its own — YOU are responsible for passing the
  history each time. LangChain just gives you the typed objects to do it cleanly.

FLOW:
  [SystemMessage, HumanMessage] ──► ChatOpenAI.invoke() ──► AIMessage
                                          │
                                    (HTTP call to
                                     OpenRouter API)
"""

import os
from dotenv import load_dotenv

from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage

load_dotenv()

# ─────────────────────────────────────────────
# 1. CREATE THE MODEL
# ─────────────────────────────────────────────
#
# ChatOpenAI normally points to api.openai.com.
# We override base_url to point at OpenRouter instead.
# The interface (invoke, stream, etc.) is IDENTICAL — that's the whole point.
#
# Key params:
#   model       — the model string OpenRouter expects
#   temperature — 0.0 = deterministic, 1.0 = creative
#   base_url    — swap provider without changing any other code
#   api_key     — OpenRouter key, passed as openai_api_key
#
model = ChatOpenAI(
    model=os.getenv("OPENROUTER_MODEL", "openai/gpt-4o-mini"),
    temperature=0.7,
    base_url="https://openrouter.ai/api/v1",
    api_key=os.getenv("OPENROUTER_API_KEY"),
)

# ─────────────────────────────────────────────
# 2. BUILD A MESSAGE LIST (the "conversation")
# ─────────────────────────────────────────────
#
# Think of this as the full context you're handing to the model.
# Order matters — it's chronological.
#
messages = [
    # SystemMessage → sets the model's persona/behavior.
    # The user never sees this; it's instructions to the model.
    SystemMessage(content="You are a helpful stock market assistant. Be concise."),

    # HumanMessage → what the user said.
    HumanMessage(content="What is a stock ticker symbol? Give me one example."),
]

# ─────────────────────────────────────────────
# 3. CALL THE MODEL
# ─────────────────────────────────────────────
#
# .invoke() sends the messages and blocks until the response arrives.
# It returns an AIMessage object, not a raw string.
#
print("\n--- Sending messages to model ---")
print(f"  Messages sent: {len(messages)}")
for m in messages:
    print(f"  [{type(m).__name__}]: {m.content[:60]}...")

response: AIMessage = model.invoke(messages)

# ─────────────────────────────────────────────
# 4. INSPECT THE RESPONSE
# ─────────────────────────────────────────────
#
# response.content      → the text answer (str)
# response.response_metadata → token usage, model name, finish reason
# response.id           → unique message ID
#
print("\n--- Response ---")
print(f"  Type     : {type(response).__name__}")
print(f"  Content  : {response.content}")
print(f"  Tokens   : {response.response_metadata.get('token_usage', 'N/A')}")

# ─────────────────────────────────────────────
# 5. SIMULATING MULTI-TURN CONVERSATION
# ─────────────────────────────────────────────
#
# To continue the conversation, append the AIMessage to the list,
# then add a new HumanMessage, then call invoke() again.
# This is exactly what a memory/history system automates later.
#
messages.append(response)  # AIMessage from the model
messages.append(HumanMessage(content="Give me another example ticker, but from Europe."))

print("\n--- Continuing conversation (turn 2) ---")
print(f"  Messages sent: {len(messages)}")

response2: AIMessage = model.invoke(messages)

print(f"  Response: {response2.content}")

# ─────────────────────────────────────────────
# SUMMARY
# ─────────────────────────────────────────────
#
# What just happened:
#
#  Turn 1:
#    [System, Human] → invoke() → AIMessage
#
#  Turn 2:
#    [System, Human, AIMessage, Human] → invoke() → AIMessage
#
#  The model "remembers" turn 1 only because we included it in the list.
#  Remove the AIMessage and it would have no context.
#
# Next: m2_prompt_templates.py — stop hardcoding strings, use templates.
