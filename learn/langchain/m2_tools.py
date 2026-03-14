"""
Module 2.1 + 2.2 — Defining Tools & Binding Them to the Model
===============================================================

WHAT THIS FILE TEACHES:
  - @tool decorator: simplest way to define a tool
  - Tool anatomy: name, description, args_schema
  - StructuredTool: more control, explicit Pydantic schema
  - model.bind_tools([...]): how the model learns tools exist
  - Inspecting raw tool_calls on AIMessage (before agents automate this)
  - Manually executing a tool call and sending a ToolMessage back

THE CORE INSIGHT:
  The model never calls your function. It returns a structured
  instruction saying "please call this function with these args".
  YOU execute it and send the result back as a ToolMessage.
  An agent (Module 3) automates this loop.

TOOL CALL CYCLE (manual version):
  bind_tools → model.invoke → AIMessage(tool_calls=[...])
       → execute function → ToolMessage → model.invoke → final AIMessage
"""

import os
from dotenv import load_dotenv
from typing import Optional
from pydantic import BaseModel, Field

from langchain_openai import ChatOpenAI
from langchain_core.tools import tool, StructuredTool
from langchain_core.messages import HumanMessage, ToolMessage, SystemMessage

load_dotenv()

model = ChatOpenAI(
    model=os.getenv("OPENROUTER_MODEL", "openai/gpt-4o-mini"),
    temperature=0,
    base_url="https://openrouter.ai/api/v1",
    api_key=os.getenv("OPENROUTER_API_KEY"),
)

# ─────────────────────────────────────────────
# DEFINING TOOLS
# ─────────────────────────────────────────────

# ── Method 1: @tool decorator ────────────────────────────────────────────────
#
# Rules:
#   - Function name       → tool name (the model uses this to call it)
#   - Docstring           → description (the model reads this to decide when to call it)
#   - Type annotations    → argument types (the model fills these)
#   - Return type         → what the tool gives back (always returned as string to the model)
#
# The docstring is critical. If it's vague, the model won't know when to use the tool.
#
@tool
def get_stock_price(ticker: str) -> str:
    """
    Get the current market price of a stock.
    Use this when the user asks about the price, value, or current quote of a stock.

    Args:
        ticker: The stock ticker symbol, e.g. AAPL, MSFT, NVDA
    """
    # In production: call the real Polygon API here.
    # For now, a stub so we can focus on the tool calling mechanism.
    prices = {"AAPL": 189.50, "MSFT": 415.20, "NVDA": 875.40, "TSLA": 245.10}
    price = prices.get(ticker.upper())
    if price is None:
        return f"Price for {ticker} not found."
    return f"{ticker.upper()}: ${price:.2f}"


@tool
def get_company_info(ticker: str) -> str:
    """
    Get basic company information for a stock ticker.
    Use this when the user asks what a company does, its sector, or its description.

    Args:
        ticker: The stock ticker symbol, e.g. AAPL, MSFT, NVDA
    """
    companies = {
        "AAPL": "Apple Inc. — Consumer electronics & software. Sector: Technology.",
        "MSFT": "Microsoft Corp. — Cloud computing & software. Sector: Technology.",
        "NVDA": "NVIDIA Corp. — GPUs & AI accelerators. Sector: Semiconductors.",
        "TSLA": "Tesla Inc. — Electric vehicles & energy. Sector: Automotive.",
    }
    info = companies.get(ticker.upper())
    if info is None:
        return f"Company info for {ticker} not found."
    return info


# ── Method 2: StructuredTool with explicit Pydantic schema ───────────────────
#
# Use this when:
#   - You need multiple optional arguments
#   - You want field-level descriptions (more precise than docstring alone)
#   - You need Pydantic validation on the inputs
#
class StockComparisonInput(BaseModel):
    ticker_a: str = Field(description="First stock ticker symbol, e.g. AAPL")
    ticker_b: str = Field(description="Second stock ticker symbol, e.g. MSFT")
    metric: Optional[str] = Field(
        default="price",
        description="What to compare: 'price', 'sector'. Defaults to price."
    )

def _compare_stocks(ticker_a: str, ticker_b: str, metric: str = "price") -> str:
    """Core logic — kept separate from the tool definition."""
    prices = {"AAPL": 189.50, "MSFT": 415.20, "NVDA": 875.40, "TSLA": 245.10}
    if metric == "price":
        a = prices.get(ticker_a.upper(), "N/A")
        b = prices.get(ticker_b.upper(), "N/A")
        return f"Price comparison — {ticker_a}: ${a} vs {ticker_b}: ${b}"
    return f"Metric '{metric}' not supported."

compare_stocks = StructuredTool.from_function(
    func=_compare_stocks,
    name="compare_stocks",
    description=(
        "Compare two stocks side by side. "
        "Use when the user asks to compare, contrast, or evaluate two stocks against each other."
    ),
    args_schema=StockComparisonInput,
)


# ── Inspect the tools ─────────────────────────────────────────────────────────
#
# Each tool has: .name, .description, .args_schema
# These are what get sent to the model as function definitions.
#
print("\n" + "="*55)
print("TOOL DEFINITIONS")
print("="*55)

for t in [get_stock_price, get_company_info, compare_stocks]:
    print(f"\n  name        : {t.name}")
    print(f"  description : {t.description[:70]}...")
    print(f"  schema      : {t.args_schema.model_json_schema()['properties']}")


# ─────────────────────────────────────────────
# BINDING TOOLS TO THE MODEL
# ─────────────────────────────────────────────
#
# .bind_tools() attaches tool schemas to the model.
# Every call to model_with_tools.invoke() will include the tool definitions.
# The model can then choose to call one (or none).
#
tools = [get_stock_price, get_company_info, compare_stocks]
model_with_tools = model.bind_tools(tools)


# ─────────────────────────────────────────────
# EXAMPLE 1: Model calls a tool
# ─────────────────────────────────────────────
#
# When the model decides to use a tool, it returns an AIMessage where:
#   content   = ""              (empty — it's not giving a text answer)
#   tool_calls = [...]          (list of tool call instructions)
#
print("\n" + "="*55)
print("EXAMPLE 1: Model decides to call a tool")
print("="*55)

messages = [HumanMessage("What is the current price of Apple stock?")]
response = model_with_tools.invoke(messages)

print(f"\n  content    : '{response.content}'")       # empty
print(f"  tool_calls : {response.tool_calls}")

# ─────────────────────────────────────────────
# EXAMPLE 2: Model answers without a tool
# ─────────────────────────────────────────────
#
# The model only calls tools when it thinks it needs to.
# A general knowledge question doesn't need a tool.
#
print("\n" + "="*55)
print("EXAMPLE 2: Model answers without a tool")
print("="*55)

messages = [HumanMessage("What does P/E ratio mean?")]
response = model_with_tools.invoke(messages)

print(f"\n  content    : {response.content}")
print(f"  tool_calls : {response.tool_calls}")        # empty list []


# ─────────────────────────────────────────────
# EXAMPLE 3: Manually executing the full tool call cycle
# ─────────────────────────────────────────────
#
# This is what an agent automates. We do it manually here to understand
# exactly what's happening step by step.
#
# STEP 1: model signals it wants to call a tool
# STEP 2: we find the right function and call it
# STEP 3: we wrap the result in a ToolMessage and send it back
# STEP 4: model produces the final human-readable answer
#
print("\n" + "="*55)
print("EXAMPLE 3: Full manual tool call cycle")
print("="*55)

# Tool registry — maps tool name → callable
tool_registry = {t.name: t for t in tools}

# STEP 1 — first model call
messages = [
    SystemMessage("You are a stock assistant. Use tools to get real data."),
    HumanMessage("Compare the price of AAPL and NVDA for me."),
]
ai_response = model_with_tools.invoke(messages)

print(f"\n  STEP 1 — AIMessage:")
print(f"    content    : '{ai_response.content}'")
print(f"    tool_calls : {ai_response.tool_calls}")

# STEP 2 — execute each tool call
messages.append(ai_response)   # add AIMessage to history

tool_messages = []
for call in ai_response.tool_calls:
    tool_fn  = tool_registry[call["name"]]
    result   = tool_fn.invoke(call["args"])          # execute the tool

    tool_msg = ToolMessage(
        content      = result,
        tool_call_id = call["id"],                   # must match the call ID
    )
    tool_messages.append(tool_msg)
    print(f"\n  STEP 2 — Tool executed:")
    print(f"    tool       : {call['name']}")
    print(f"    args       : {call['args']}")
    print(f"    result     : {result}")

messages.extend(tool_messages)  # add ToolMessages to history

# STEP 3 — second model call with tool results
final_response = model_with_tools.invoke(messages)

print(f"\n  STEP 3 — Final AIMessage:")
print(f"    content    : {final_response.content}")
print(f"    tool_calls : {final_response.tool_calls}")  # empty — done

# ─────────────────────────────────────────────
# SUMMARY
# ─────────────────────────────────────────────
#
# @tool                    decorator → name from fn, description from docstring
# StructuredTool           explicit schema → better for complex/optional args
# model.bind_tools([...])  attach tool schemas to model
#
# When model calls a tool:
#   AIMessage.content    = ""         (empty)
#   AIMessage.tool_calls = [{name, args, id}]
#
# You execute the tool, wrap result in ToolMessage(content, tool_call_id)
# Send it back → model gives the final answer
#
# Next: m2_mcp_adapter.py — replace stub functions with real MCP tools
