"""
Module 1.3 — LCEL & the | Pipe Operator
=========================================

WHAT THIS FILE TEACHES:
  - The Runnable interface (why | works)
  - Building chains with | (template | model | parser)
  - StrOutputParser — extracting the text from AIMessage
  - RunnableLambda — wrapping any function into a chain
  - RunnablePassthrough — passing input through unchanged
  - RunnableParallel — running multiple chains on the same input
  - .stream() — getting tokens as they arrive (token streaming)
  - .batch() — running multiple inputs in parallel

TYPE FLOW:
  dict → [ChatPromptTemplate] → PromptValue
       → [ChatOpenAI]         → AIMessage
       → [StrOutputParser]    → str

THE CORE IDEA:
  Every step is a Runnable. A chain is also a Runnable.
  You can compose anything that has .invoke() with |.
"""

import os
from dotenv import load_dotenv

from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableLambda, RunnablePassthrough, RunnableParallel

load_dotenv()

model = ChatOpenAI(
    model=os.getenv("OPENROUTER_MODEL", "openai/gpt-4o-mini"),
    temperature=0.7,
    base_url="https://openrouter.ai/api/v1",
    api_key=os.getenv("OPENROUTER_API_KEY"),
)

template = ChatPromptTemplate.from_messages([
    ("system", "You are a stock assistant. Be concise, max 2 sentences."),
    ("human", "{question}"),
])

# ─────────────────────────────────────────────
# EXAMPLE 1: The basic chain
# ─────────────────────────────────────────────
#
# template | model
#   → input:  dict with template variables
#   → output: AIMessage
#
# template | model | StrOutputParser()
#   → input:  dict with template variables
#   → output: str  (just the text, no AIMessage wrapper)
#
# StrOutputParser does one thing: calls response.content on the AIMessage.
# That's it. But it makes the chain output a plain string, which is
# easier to work with in UIs and downstream steps.
#
print("\n" + "="*50)
print("EXAMPLE 1: Basic chain")
print("="*50)

# Without parser → output is AIMessage
chain_no_parser = template | model
result = chain_no_parser.invoke({"question": "What is a P/E ratio?"})
print(f"\n  type (no parser) : {type(result).__name__}")
print(f"  content          : {result.content}")

# With parser → output is str
chain = template | model | StrOutputParser()
result = chain.invoke({"question": "What is a P/E ratio?"})
print(f"\n  type (with parser): {type(result).__name__}")
print(f"  content           : {result}")

# ─────────────────────────────────────────────
# EXAMPLE 2: The chain is itself a Runnable
# ─────────────────────────────────────────────
#
# Because chain is a Runnable, you can:
#   - pipe it into another chain
#   - pass it as an argument
#   - call .invoke(), .stream(), .batch() on it
#
# Here we pipe the chain output into another RunnableLambda
# just to demonstrate composability.
#
print("\n" + "="*50)
print("EXAMPLE 2: Chains are composable Runnables")
print("="*50)

# RunnableLambda wraps any Python function as a Runnable
uppercase = RunnableLambda(lambda text: text.upper())

chain_with_transform = template | model | StrOutputParser() | uppercase
result = chain_with_transform.invoke({"question": "What is a dividend?"})
print(f"\n  Result (uppercased): {result}")

# ─────────────────────────────────────────────
# EXAMPLE 3: RunnablePassthrough
# ─────────────────────────────────────────────
#
# Passes its input through unchanged.
# Seems useless alone, but critical inside RunnableParallel
# when you need to forward the original input alongside processed output.
#
print("\n" + "="*50)
print("EXAMPLE 3: RunnablePassthrough")
print("="*50)

passthrough_chain = RunnablePassthrough() | RunnableLambda(lambda x: f"Got: {x}")
result = passthrough_chain.invoke("hello")
print(f"\n  Result: {result}")

# ─────────────────────────────────────────────
# EXAMPLE 4: RunnableParallel — run two chains simultaneously
# ─────────────────────────────────────────────
#
# RunnableParallel takes the SAME input and fans it out to multiple chains.
# All branches run in parallel (threads). Output is a dict.
#
# Shape:
#          ┌─── chain_a ───► "answer_a"
#  input ──┤
#          └─── chain_b ───► "answer_b"
#
#  output: {"answer_a": ..., "answer_b": ...}
#
# Real use case: run two different analysis prompts on the same ticker
# at the same time, then combine results.
#
print("\n" + "="*50)
print("EXAMPLE 4: RunnableParallel")
print("="*50)

bullish_template = ChatPromptTemplate.from_messages([
    ("system", "You are a bullish stock analyst. One sentence only."),
    ("human", "Give a bullish take on {ticker}."),
])

bearish_template = ChatPromptTemplate.from_messages([
    ("system", "You are a bearish stock analyst. One sentence only."),
    ("human", "Give a bearish take on {ticker}."),
])

parser = StrOutputParser()

parallel_chain = RunnableParallel(
    bullish = bullish_template | model | parser,
    bearish = bearish_template | model | parser,
    ticker  = RunnablePassthrough() | RunnableLambda(lambda x: x["ticker"]),
)

result = parallel_chain.invoke({"ticker": "NVDA"})

print(f"\n  Ticker  : {result['ticker']}")
print(f"  Bullish : {result['bullish']}")
print(f"  Bearish : {result['bearish']}")

# ─────────────────────────────────────────────
# EXAMPLE 5: .stream() — token streaming
# ─────────────────────────────────────────────
#
# Instead of waiting for the full response, stream yields chunks
# as the model generates them. Each chunk is a partial AIMessage
# (or str if you have StrOutputParser).
#
# This is what powers the "typing" effect in chat UIs.
# Streamlit has st.write_stream() that consumes this directly.
#
print("\n" + "="*50)
print("EXAMPLE 5: Streaming tokens")
print("="*50)

stream_chain = template | model | StrOutputParser()

print("\n  Streaming response: ", end="", flush=True)
for chunk in stream_chain.stream({"question": "Explain market capitalization in simple terms."}):
    print(chunk, end="", flush=True)
print()  # newline after stream ends

# ─────────────────────────────────────────────
# EXAMPLE 6: .batch() — multiple inputs in parallel
# ─────────────────────────────────────────────
#
# Runs multiple inputs through the same chain concurrently.
# Returns a list of results in the same order as inputs.
#
# Useful for: processing multiple tickers at once,
# generating summaries for a watchlist, etc.
#
print("\n" + "="*50)
print("EXAMPLE 6: Batch — multiple inputs at once")
print("="*50)

questions = [
    {"question": "What does 'going long' mean?"},
    {"question": "What does 'short selling' mean?"},
    {"question": "What is a stock split?"},
]

batch_chain = template | model | StrOutputParser()
results = batch_chain.batch(questions)

for q, r in zip(questions, results):
    print(f"\n  Q: {q['question']}")
    print(f"  A: {r}")

# ─────────────────────────────────────────────
# SUMMARY
# ─────────────────────────────────────────────
#
# | operator       chains Runnables: output of left → input of right
# StrOutputParser  extracts .content from AIMessage → plain str
# RunnableLambda   wraps any function into the chain
# RunnableParallel fans out same input to multiple chains → dict output
# RunnablePassthrough  passes input unchanged (used inside parallel)
#
# Methods available on any chain:
#   .invoke(input)         → single result (blocking)
#   .stream(input)         → iterator of chunks (streaming)
#   .batch([input1, ...])  → list of results (parallel)
#   .ainvoke(input)        → async invoke (we'll use in Module 3 with LangGraph)
#
# Next: m1_output_parsers.py — beyond StrOutputParser,
#       parsing model output into structured Pydantic objects
