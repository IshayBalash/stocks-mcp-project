"""
Module 3 — LangGraph Agents
=============================

WHAT THIS FILE TEACHES:
  - create_agent: build a full agent in one line (replaces deprecated create_react_agent)
  - State: the message list that flows through the graph
  - ainvoke: run the agent (async — it may call tools N times)
  - astream_events: stream tokens + tool events as they happen
  - Connecting MCP tools to the agent (same tools from Module 2.4)

HOW THE AGENT LOOP WORKS:
  1. Agent calls the model with current messages
  2. Model returns tool_calls → agent executes them, appends results
  3. Loop back to step 1
  4. Model returns no tool_calls → agent stops, returns final state

INPUT/OUTPUT shape:
  in:  {"messages": [HumanMessage(...)]}
  out: {"messages": [Human, AI(tool_calls), Tool, AI(final), ...]}

REQUIRES: MCP server running → uv run mcp-server
"""

import os
import asyncio
from dotenv import load_dotenv

from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage
from langchain_core.tools import tool
from langchain.agents import create_agent
from langchain_mcp_adapters.client import MultiServerMCPClient

load_dotenv()

model = ChatOpenAI(
    model=os.getenv("OPENROUTER_MODEL", "openai/gpt-4o-mini"),
    temperature=0,
    base_url="https://openrouter.ai/api/v1",
    api_key=os.getenv("OPENROUTER_API_KEY"),
)

MCP_SERVER_URL = os.getenv("MCP_SERVER_URL", "http://localhost:8001/sse")


# ─────────────────────────────────────────────
# PART 1: Agent with stub tools (no MCP needed)
# ─────────────────────────────────────────────
#
# Start here to understand the agent without needing the MCP server.
# We use the same stub tools from Module 2.1.
#

@tool
def get_stock_price(ticker: str) -> str:
    """Get the current price of a stock. Use when asked about stock price or value."""
    prices = {"AAPL": 189.50, "MSFT": 415.20, "NVDA": 875.40, "TSLA": 245.10}
    price = prices.get(ticker.upper())
    return f"{ticker.upper()}: ${price:.2f}" if price else f"Price for {ticker} not found."

@tool
def get_company_info(ticker: str) -> str:
    """Get company description and sector. Use when asked what a company does."""
    companies = {
        "AAPL": "Apple Inc. — Consumer electronics & software. Sector: Technology.",
        "NVDA": "NVIDIA Corp. — GPUs & AI accelerators. Sector: Semiconductors.",
    }
    return companies.get(ticker.upper(), f"Info for {ticker} not found.")


async def part1_basic_agent():
    print("\n" + "="*55)
    print("PART 1: Basic agent with stub tools")
    print("="*55)

    # create_agent — builds the full graph in one call:
    #   model         → the LLM to use
    #   tools         → list of tools the agent can call
    #   system_prompt → optional system message (str or SystemMessage)
    #
    # Returns a compiled LangGraph — a Runnable with ainvoke, astream_events, etc.
    #
    agent = create_agent(
        model=model,
        tools=[get_stock_price, get_company_info],
        system_prompt="You are a stock assistant. Always use tools to get real data.",
    )

    # ── Example 1: single tool call ─────────────────────────────────────────
    print("\n--- Example 1: Single tool call ---")

    result = await agent.ainvoke({
        "messages": [HumanMessage("What is the current price of NVDA?")]
    })

    # result["messages"] contains the full conversation the agent had internally
    print(f"\n  Messages in final state: {len(result['messages'])}")
    for msg in result["messages"]:
        role = type(msg).__name__
        content = msg.content[:80] if msg.content else f"[tool_calls: {getattr(msg, 'tool_calls', [])}]"
        print(f"  [{role}]: {content}")

    # The final answer is always the last message
    print(f"\n  Final answer: {result['messages'][-1].content}")

    # ── Example 2: multi-tool call ───────────────────────────────────────────
    #
    # The model may decide it needs multiple tools — either in one round
    # (parallel tool calls) or across multiple rounds.
    #
    print("\n--- Example 2: Multi-tool, model decides the sequence ---")

    result = await agent.ainvoke({
        "messages": [HumanMessage("What does AAPL do and what is its current price?")]
    })

    print(f"\n  Messages in final state: {len(result['messages'])}")
    for msg in result["messages"]:
        role = type(msg).__name__
        content = msg.content[:80] if msg.content else f"[tool_calls: {getattr(msg, 'tool_calls', [])}]"
        print(f"  [{role}]: {content}")

    print(f"\n  Final answer: {result['messages'][-1].content}")

    # ── Example 3: no tool needed ────────────────────────────────────────────
    #
    # The agent only calls tools when it thinks it needs to.
    # General knowledge questions are answered directly.
    #
    print("\n--- Example 3: No tool needed ---")

    result = await agent.ainvoke({
        "messages": [HumanMessage("What does P/E ratio mean?")]
    })

    print(f"\n  Messages in state: {len(result['messages'])}  (just 2 — no tool calls)")
    print(f"  Final answer: {result['messages'][-1].content}")


# ─────────────────────────────────────────────
# PART 2: Streaming — tokens + tool events
# ─────────────────────────────────────────────
#
# astream_events() yields events as they happen:
#   - "on_chat_model_stream"   → token by token (the typing effect)
#   - "on_tool_start"          → tool is about to be called
#   - "on_tool_end"            → tool finished, here's the result
#
# This is what powers live UI updates in Streamlit.
#
async def part2_streaming():
    print("\n" + "="*55)
    print("PART 2: Streaming events")
    print("="*55)

    agent = create_agent(
        model=model,
        tools=[get_stock_price, get_company_info],
        system_prompt="You are a stock assistant. Always use tools to get real data.",
    )

    print("\n  Streaming response for: 'What is AAPL's price?'\n")

    async for event in agent.astream_events(
        {"messages": [HumanMessage("What is AAPL's price?")]},
        version="v2",
    ):
        kind = event["event"]

        if kind == "on_tool_start":
            print(f"\n  [TOOL CALLED] {event['name']} with args: {event['data'].get('input')}")

        elif kind == "on_tool_end":
            print(f"  [TOOL RESULT] {event['data'].get('output')}")

        elif kind == "on_chat_model_stream":
            chunk = event["data"]["chunk"].content
            if chunk:
                print(chunk, end="", flush=True)

    print("\n")


# ─────────────────────────────────────────────
# PART 3: Agent with real MCP tools
# ─────────────────────────────────────────────
#
# Identical to Part 1, but tools come from the MCP server.
# The agent doesn't know or care — same interface.
#
# REQUIRES: uv run mcp-server (in another terminal)
#
async def part3_mcp_agent():
    print("\n" + "="*55)
    print("PART 3: Agent with real MCP tools")
    print("="*55)

    client = MultiServerMCPClient({
        "stocks": {"url": MCP_SERVER_URL, "transport": "sse"}
    })
    tools = await client.get_tools()

    print(f"\n  Loaded {len(tools)} tools from MCP server: {[t.name for t in tools]}")

    agent = create_agent(
        model=model,
        tools=tools,
        system_prompt="You are a stock assistant. Always use tools to get real data.",
    )

    result = await agent.ainvoke({
        "messages": [HumanMessage("What was the last closing price of AAPL?")]
    })

    print(f"\n  Final answer: {result['messages'][-1].content}")


# ─────────────────────────────────────────────
# Entry point — run parts 1 and 2 by default
# Change to include part3 once MCP server is running
# ─────────────────────────────────────────────
async def main():
    await part1_basic_agent()
    await part2_streaming()
    # await part3_mcp_agent()    # ← uncomment when MCP server is running


if __name__ == "__main__":
    asyncio.run(main())