"""
Module 1.4 — Output Parsers & Structured Output
=================================================

WHAT THIS FILE TEACHES:
  - StrOutputParser        — extract plain text from AIMessage
  - JsonOutputParser       — parse JSON text from model output (old way)
  - PydanticOutputParser   — parse into typed Pydantic model (old way)
  - with_structured_output — use model's native tool calling (new way, preferred)

WHY THIS MATTERS FOR OUR PROJECT:
  When the user types "Tell me about Apple in the US market", we need to
  extract structured data (ticker="AAPL", market="US") to fill our templates
  and call our MCP tools. with_structured_output() is how we do that reliably.

OLD WAY flow:
  prompt (with format instructions injected) → model → text → parser → object

NEW WAY flow:
  prompt → model.with_structured_output(Schema) → Schema instance
                    (tool calling under the hood, guaranteed structure)
"""

import os
from typing import Optional
from dotenv import load_dotenv
from pydantic import BaseModel, Field

from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser, JsonOutputParser
from langchain_core.output_parsers import PydanticOutputParser

load_dotenv()

model = ChatOpenAI(
    model=os.getenv("OPENROUTER_MODEL", "openai/gpt-4o-mini"),
    temperature=0,          # 0 = deterministic, important for structured output
    base_url="https://openrouter.ai/api/v1",
    api_key=os.getenv("OPENROUTER_API_KEY"),
)

# ─────────────────────────────────────────────
# EXAMPLE 1: StrOutputParser (already familiar)
# ─────────────────────────────────────────────
#
# Recap: just extracts AIMessage.content as a plain string.
# The simplest parser — use when you just need the text.
#
print("\n" + "="*55)
print("EXAMPLE 1: StrOutputParser (recap)")
print("="*55)

chain = (
    ChatPromptTemplate.from_messages([("human", "{question}")])
    | model
    | StrOutputParser()
)
result = chain.invoke({"question": "What does 'bull market' mean? One sentence."})
print(f"\n  type   : {type(result).__name__}")   # str
print(f"  result : {result}")

# ─────────────────────────────────────────────
# EXAMPLE 2: JsonOutputParser (old way)
# ─────────────────────────────────────────────
#
# Asks the model to return JSON text, then parses it into a dict.
#
# Important: YOU must tell the model to return JSON in your prompt.
# The parser only handles the parsing step — it does NOT instruct the model.
# If the model returns text instead of JSON, this crashes.
#
# This is the fragility of the old approach.
#
print("\n" + "="*55)
print("EXAMPLE 2: JsonOutputParser (old way)")
print("="*55)

json_template = ChatPromptTemplate.from_messages([
    ("system", "You are a data extractor. Always respond with valid JSON only. No explanation."),
    ("human", "Extract the stock ticker and company name from: '{text}'\n\nReturn JSON with keys: ticker, company_name"),
])

json_chain = json_template | model | JsonOutputParser()

result = json_chain.invoke({"text": "I want to invest in Microsoft"})
print(f"\n  type          : {type(result).__name__}")   # dict
print(f"  ticker        : {result.get('ticker')}")
print(f"  company_name  : {result.get('company_name')}")

# ─────────────────────────────────────────────
# EXAMPLE 3: PydanticOutputParser (old way, typed)
# ─────────────────────────────────────────────
#
# Like JsonOutputParser but returns a typed Pydantic object instead of a dict.
# Also injects format instructions into the prompt automatically.
#
# Still the old approach — relies on model following text instructions.
#
print("\n" + "="*55)
print("EXAMPLE 3: PydanticOutputParser (old way)")
print("="*55)

class StockMention(BaseModel):
    ticker: str = Field(description="The stock ticker symbol, e.g. AAPL")
    company_name: str = Field(description="The full company name")
    sentiment: str = Field(description="User sentiment: bullish, bearish, or neutral")

pydantic_parser = PydanticOutputParser(pydantic_object=StockMention)

# Note: {format_instructions} is injected by the parser — it tells the model
# exactly what JSON schema to produce. This is what makes old-way parsers fragile:
# it depends on the model following a long text instruction.
pydantic_template = ChatPromptTemplate.from_messages([
    ("system", "You extract structured data from user messages.\n{format_instructions}"),
    ("human", "{text}"),
]).partial(format_instructions=pydantic_parser.get_format_instructions())

pydantic_chain = pydantic_template | model | pydantic_parser

result = pydantic_chain.invoke({"text": "I'm very bullish on Tesla right now"})
print(f"\n  type         : {type(result).__name__}")    # StockMention
print(f"  ticker       : {result.ticker}")
print(f"  company_name : {result.company_name}")
print(f"  sentiment    : {result.sentiment}")

# See what format_instructions looks like — this is injected into the prompt
print(f"\n  (format_instructions preview):")
print(f"  {pydantic_parser.get_format_instructions()[:200]}...")

# ─────────────────────────────────────────────
# EXAMPLE 4: with_structured_output (new way — use this)
# ─────────────────────────────────────────────
#
# This is the modern approach. Instead of instructing the model in text,
# it uses the model's native tool/function calling to fill a schema.
#
# The model never produces free text — it fills the schema fields directly.
# No format instructions in the prompt. No JSON parsing. No crash risk.
#
# How it works under the hood:
#   1. LangChain converts your Pydantic schema into a tool definition
#   2. Sends it to the model as a function/tool the model MUST call
#   3. The model fills the function arguments (your schema fields)
#   4. LangChain converts the arguments back into your Pydantic object
#
print("\n" + "="*55)
print("EXAMPLE 4: with_structured_output (new way, preferred)")
print("="*55)

class StockQuery(BaseModel):
    """Structured representation of a user's stock query."""
    ticker: str = Field(description="The stock ticker symbol, uppercase. E.g. AAPL, MSFT, NVDA")
    company_name: str = Field(description="The full company name")
    market: str = Field(description="The stock market. E.g. US, EU, NASDAQ")
    intent: str = Field(description="What the user wants: price, analysis, news, comparison")

# .with_structured_output() wraps the model — same interface, typed output
structured_model = model.with_structured_output(StockQuery)

# Notice: no format instructions in the prompt. Clean and simple.
extraction_template = ChatPromptTemplate.from_messages([
    ("system", "You extract structured information from user stock queries."),
    ("human", "{user_input}"),
])

extraction_chain = extraction_template | structured_model

# Test with various natural language inputs
test_inputs = [
    "What's the current price of Apple?",
    "Give me an analysis of Nvidia's performance",
    "I want to compare Microsoft and Google",
]

for user_input in test_inputs:
    result = extraction_chain.invoke({"user_input": user_input})
    print(f"\n  Input        : {user_input}")
    print(f"  ticker       : {result.ticker}")
    print(f"  company_name : {result.company_name}")
    print(f"  market       : {result.market}")
    print(f"  intent       : {result.intent}")
    print(f"  type         : {type(result).__name__}")   # StockQuery ← real Pydantic object

# ─────────────────────────────────────────────
# EXAMPLE 5: Optional fields
# ─────────────────────────────────────────────
#
# Fields can be Optional — the model fills them when present,
# leaves them None when the information isn't in the input.
#
print("\n" + "="*55)
print("EXAMPLE 5: Optional fields with structured output")
print("="*55)

class StockQueryOptional(BaseModel):
    """Stock query where some fields may not be present in user input."""
    ticker: str = Field(description="Stock ticker symbol")
    timeframe: Optional[str] = Field(
        default=None,
        description="Time period if mentioned: today, this week, this month, this year"
    )
    comparison_ticker: Optional[str] = Field(
        default=None,
        description="Second ticker if the user is comparing two stocks"
    )

structured_model_2 = model.with_structured_output(StockQueryOptional,strict=True)

result_a = structured_model_2.invoke("What's AAPL doing this week?")
result_b = structured_model_2.invoke("How is Tesla performing?")
result_c = structured_model_2.invoke("Compare AAPL vs MSFT this month")

print(f"\n  'AAPL this week'   → ticker={result_a.ticker}, timeframe={result_a.timeframe}, compare={result_a.comparison_ticker}")
print(f"  'Tesla performing' → ticker={result_b.ticker}, timeframe={result_b.timeframe}, compare={result_b.comparison_ticker}")
print(f"  'AAPL vs MSFT'     → ticker={result_c.ticker}, timeframe={result_c.timeframe}, compare={result_c.comparison_ticker}")

# ─────────────────────────────────────────────
# SUMMARY
# ─────────────────────────────────────────────
#
# Parser                    Output type    Reliability    Use when
# ─────────────────────────────────────────────────────────────────────
# StrOutputParser           str            high           just need the text
# JsonOutputParser          dict           medium         need dict, no schema
# PydanticOutputParser      Pydantic obj   medium         typed, but old way
# with_structured_output()  Pydantic obj   high           always prefer this
#
# Rule: use with_structured_output() by default.
# Only fall back to parsers if the model doesn't support tool calling.
#
# Next: m2_tools.py — defining tools with @tool and binding them to the model.
# with_structured_output is actually the same mechanism as tool calling —
# so you already understand how tools work under the hood.
