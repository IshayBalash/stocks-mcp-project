"""
Module 2.4 — MCP Adapter
==========================

WHAT THIS FILE TEACHES:
  - MultiServerMCPClient: connect to one or more MCP servers
  - client.get_tools(): get all MCP tools as LangChain BaseTool objects
  - async/await: why MCP requires async (network calls)
  - Inspecting the wrapped tools — same interface as @tool
  - Manually running the tool call cycle with real MCP tools

IMPORTANT: The MCP server must be running before you run this file.
  Terminal 1: uv run mcp-server

WHY ASYNC?
  The MCP adapter talks to a server over SSE (network).
  Network I/O is async — while waiting for the server to respond,
  Python can do other things instead of blocking.

  async def   → marks a function as async (it can use await inside)
  await       → "wait for this, but don't block the whole program"
  asyncio.run → entry point: run an async function from normal sync code

THE PATTERN (always the same):
  client = MultiServerMCPClient({...})
  tools  = await client.get_tools()    ← List[BaseTool], same as @tool objects
"""

import os
import asyncio
from dotenv import load_dotenv

from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage, ToolMessage
from langchain_mcp_adapters.client import MultiServerMCPClient

load_dotenv()

model = ChatOpenAI(
    model=os.getenv("OPENROUTER_MODEL", "openai/gpt-4o-mini"),
    temperature=0,
    base_url="https://openrouter.ai/api/v1",
    api_key=os.getenv("OPENROUTER_API_KEY"),
)

MCP_SERVER_URL = os.getenv("MCP_SERVER_URL")


async def main():

    # ─────────────────────────────────────────────
    # STEP 1: Connect to MCP server & get tools
    # ─────────────────────────────────────────────
    #
    # Create client and await get_tools() — this connects to the server,
    # fetches all tool definitions, and wraps them as LangChain tools.
    #
    # You can list multiple servers:
    #   {"stocks": {...}, "news": {...}, "airbnb": {...}}
    # All their tools get merged into one flat list.
    #
    client = MultiServerMCPClient({
        "stocks": {
            "url": MCP_SERVER_URL,
            "transport": "sse",
        }
    })
    tools = await client.get_tools()

    # ─────────────────────────────────────────────
    # STEP 2: Inspect the tools
    # ─────────────────────────────────────────────
    #
    # These are real LangChain BaseTool objects — same as @tool.
    # The adapter translated MCP tool schemas into LangChain tool schemas.
    # The model can't tell the difference.
    #
    print("\n" + "="*55)
    print("MCP TOOLS (wrapped as LangChain tools)")
    print("="*55)

    for t in tools:
        print(f"\n  name        : {t.name}")
        print(f"  description : {t.description[:80]}...")

    # ─────────────────────────────────────────────
    # STEP 3: Bind to model — identical to Module 2.1
    # ─────────────────────────────────────────────
    #
    # Same .bind_tools() call as before.
    # The model doesn't know these came from an MCP server.
    #
    model_with_tools = model.bind_tools(tools)

    # ─────────────────────────────────────────────
    # EXAMPLE 1: Model decides which MCP tool to call
    # ─────────────────────────────────────────────
    print("\n" + "="*55)
    print("EXAMPLE 1: Model decides which MCP tool to call")
    print("="*55)

    messages = [
        SystemMessage("You are a stock assistant. Use tools to get real data."),
        HumanMessage("What was the last closing price of AAPL?"),
    ]

    ai_response = model_with_tools.invoke(messages)

    print(f"\n  content    : '{ai_response.content}'")
    print(f"  tool_calls : {ai_response.tool_calls}")

    # ─────────────────────────────────────────────
    # EXAMPLE 2: Full manual cycle with real MCP data
    # ─────────────────────────────────────────────
    #
    # MCP tools are async — we use "await tool.ainvoke()"
    # instead of "tool.invoke()" (the async version).
    #
    print("\n" + "="*55)
    print("EXAMPLE 2: Full tool call cycle with real MCP data")
    print("="*55)

    tool_registry = {t.name: t for t in tools}

    messages = [
        SystemMessage("You are a stock assistant. Use tools to get real data."),
        HumanMessage("What was the last closing price of AAPL?"),
    ]

    # Turn 1 — model signals tool call
    ai_response = model_with_tools.invoke(messages)
    messages.append(ai_response)

    print(f"\n  STEP 1 — tool_calls: {ai_response.tool_calls}")

    # Execute each tool call
    tool_messages = []
    for call in ai_response.tool_calls:
        tool_fn = tool_registry[call["name"]]
        result  = await tool_fn.ainvoke(call["args"])   # ← async: goes to MCP server

        tool_msg = ToolMessage(
            content      = str(result),
            tool_call_id = call["id"],
        )
        tool_messages.append(tool_msg)

        print(f"\n  STEP 2 — tool executed:")
        print(f"    tool   : {call['name']}")
        print(f"    args   : {call['args']}")
        print(f"    result : {result}")

    messages.extend(tool_messages)

    # Turn 2 — model produces final answer with real data
    final = model_with_tools.invoke(messages)

    print(f"\n  STEP 3 — Final answer:")
    print(f"    {final.content}")


# ─────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────
#
# asyncio.run() is the bridge between sync and async.
# It starts the async event loop and runs main() inside it.
# You only ever have ONE asyncio.run() per program.
#
if __name__ == "__main__":
    asyncio.run(main())
