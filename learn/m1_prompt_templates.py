"""
Module 1.2 — Prompt Templates
==============================

WHAT THIS FILE TEACHES:
  - ChatPromptTemplate: building reusable, parameterized message lists
  - Template variables: {placeholder} syntax
  - MessagesPlaceholder: injecting dynamic message lists (history slot)
  - The two-step flow: template.invoke() → messages → model.invoke() → AIMessage

WHY TEMPLATES?
  Separating the "shape" of a prompt from its "values" means:
    - One template, many inputs (reusable)
    - Easy to test by inspecting filled messages before sending
    - Clean place to manage your system prompt as the app grows
    - Foundation for the | pipe operator in Module 1.3

FLOW:
  ChatPromptTemplate
       │ .invoke({"var": value, ...})
       ▼
  List[Message]          ← you can inspect this before sending!
       │ model.invoke(messages)
       ▼
  AIMessage
"""

import os
from dotenv import load_dotenv

from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import HumanMessage, AIMessage

load_dotenv()

model = ChatOpenAI(
    model=os.getenv("OPENROUTER_MODEL", "openai/gpt-4o-mini"),
    temperature=0.7,
    base_url="https://openrouter.ai/api/v1",
    api_key=os.getenv("OPENROUTER_API_KEY"),
)

# ─────────────────────────────────────────────
# EXAMPLE 1: Basic template with variables
# ─────────────────────────────────────────────
#
# .from_messages() takes a list of tuples: (role, template_string)
# Roles: "system", "human", "ai"  (shorthand for the Message classes)
# Variables: {curly_braces} — filled at .invoke() time
#
print("\n" + "="*50)
print("EXAMPLE 1: Basic template")
print("="*50)

basic_template = ChatPromptTemplate.from_messages([
    ("system", "You are a stock market assistant specializing in {market} stocks. Be concise."),
    ("human", "What should I know about the ticker {ticker}?"),
])

# Step 1: fill the template → get a list of messages
# Nothing is sent to the model yet. This is pure local computation.
filled_messages = basic_template.invoke({
    "market": "US",
    "ticker": "NVDA",
})

print("\n--- Template filled (before sending to model) ---")
# filled_messages.to_messages() converts the result to a plain List[BaseMessage]
for msg in filled_messages.to_messages():
    print(f"  [{type(msg).__name__}]: {msg.content}")

# Step 2: send to the model
response = model.invoke(filled_messages)
print(f"\n--- Model response ---")
print(f"  {response.content}")

# ─────────────────────────────────────────────
# EXAMPLE 2: Reusing the same template
# ─────────────────────────────────────────────
#
# The template is defined once. The values change.
# This is the core benefit — decouple structure from data.
#
print("\n" + "="*50)
print("EXAMPLE 2: Same template, different input")
print("="*50)

filled_messages_2 = basic_template.invoke({
    "market": "European",
    "ticker": "ASML",
})

response_2 = model.invoke(filled_messages_2)
print(f"\n  Response about ASML: {response_2.content}")

# ─────────────────────────────────────────────
# EXAMPLE 3: MessagesPlaceholder — history slot
# ─────────────────────────────────────────────
#
# MessagesPlaceholder("key") reserves a slot in the template.
# At invoke() time, you pass a list of messages under that key.
# The list gets spliced into the message list at that position.
#
# This is the foundation of conversation memory:
#   - Redis/Postgres loads past messages
#   - They get injected here
#   - The model sees the full context
#
print("\n" + "="*50)
print("EXAMPLE 3: MessagesPlaceholder (history slot)")
print("="*50)

history_template = ChatPromptTemplate.from_messages([
    ("system", "You are a stock assistant. Answer based on context."),
    MessagesPlaceholder("history"),   # ← slot for past messages
    ("human", "{question}"),          # ← current question
])

# Simulate a past conversation that would come from a database
simulated_history = [
    HumanMessage(content="What does P/E ratio mean?"),
    AIMessage(content="P/E ratio (Price-to-Earnings) measures how much investors pay per dollar of earnings. A high P/E suggests growth expectations."),
]

# Now ask a follow-up question that references the history
filled = history_template.invoke({
    "history":  simulated_history,
    "question": "Is a P/E of 35 considered high?",   # references the prior context
})

print("\n--- Full message list sent to model ---")
for msg in filled.to_messages():
    print(f"  [{type(msg).__name__}]: {msg.content[:80]}")

response_3 = model.invoke(filled)
print(f"\n--- Model response (uses history context) ---")
print(f"  {response_3.content}")

# ─────────────────────────────────────────────
# EXAMPLE 4: Inspecting the template itself
# ─────────────────────────────────────────────
#
# Useful for debugging — see what variables a template expects
# without running it.
#
print("\n" + "="*50)
print("EXAMPLE 4: Template introspection")
print("="*50)

print(f"\n  basic_template input variables : {basic_template.input_variables}")
print(f"  history_template input variables: {history_template.input_variables}")

# ─────────────────────────────────────────────
# SUMMARY
# ─────────────────────────────────────────────
#
# ChatPromptTemplate.from_messages([...])
#   → defines the shape (roles + variables)
#
# template.invoke({"var": value})
#   → fills variables → returns PromptValue (call .to_messages() to get List[Message])
#
# MessagesPlaceholder("key")
#   → reserved slot for a dynamic list of messages
#   → used for injecting history in Module 4
#
# Next: m1_lcel.py — chain template | model into one expression with the pipe operator
