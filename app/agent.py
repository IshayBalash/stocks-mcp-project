import os
import asyncio
from dotenv import load_dotenv
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI
from langchain.agents import create_agent
from langchain_mcp_adapters.client import MultiServerMCPClient
from langchain_core.messages import HumanMessage

load_dotenv()

model = ChatOpenAI(
    model=os.getenv("OPENROUTER_MODEL", "openai/gpt-4o-mini"),
    temperature=0.5,
    base_url="https://openrouter.ai/api/v1",
    api_key=os.getenv("OPENROUTER_API_KEY"),
    extra_body={"provider": {"ignore": ["Venice"]}},
)
MCP_SERVER_URL = os.getenv("MCP_SERVER_URL", "http://localhost:8001/sse")
MCP_PROMTES=[
    {
        "name": "portfolio_activity_summary",
        "description": (
            "Use ONLY when the user explicitly asks for a full portfolio summary, "
            "performance report, or deep analysis of their trading history. "
            "Do NOT use for simple data lookups like listing holdings, "
            "fetching stock prices, or reading transactions."
        )
    },
]


def make_prompt_tool(client, name: str, description: str):
    @tool(f"load_{name}", description=description)
    async def prompt_tool() -> str:
        messages = await client.get_prompt("stocks", name)
        return "\n".join(m.content for m in messages)
    return prompt_tool




async def init_agent():
    """Initialize MCP client, load tools, build agent. Call once at startup."""
    client = MultiServerMCPClient({
        "stocks": {"url": MCP_SERVER_URL, "transport": "sse"}
    })
    tools = await client.get_tools()
    tools += [make_prompt_tool(client, p['name'], p['description']) for p in MCP_PROMTES]

    ## logging to see the tools:
    if not tools:
        print("No tools found. Check MCP server connection and configuration.")
    print("------  List of available tools: -------------------")
    for t in tools:
        print(f"Tool: {t.name}")
    print("------------------------------------")

    agent = create_agent(
        model=model,
        tools=tools,
        system_prompt="You are a stock assistant. Always use tools to get real data.",
    )
    return agent


async def astream_response(agent, messages: list):
    """
    Stream agent events. Yields (event_type, data) tuples:
      ("tool_start", tool_name)
      ("tool_end",   tool_name)
      ("token",      chunk_str)
    """
    async for event in agent.astream_events({"messages": messages}, version="v2"):
        kind = event["event"]
        if kind == "on_tool_start":
            print(f"[TOOL CALL] → {event['name']}")
            yield "tool_start", event["name"]
        elif kind == "on_tool_end":
            print(f"[TOOL DONE] ← {event['name']})") 
            yield "tool_end", event["name"]
        elif kind == "on_chat_model_stream":
            chunk = event["data"]["chunk"].content
            if chunk:
                yield "token", chunk


# ── Dev runner ────────────────────────────────────────────────────────────────
async def _dev_run():
    agent = await init_agent()
    messages = [HumanMessage("read my transactions and list the companies I have invested in, for the top 3 holding show me the last closing price")]

    async for event_type, data in astream_response(agent, messages):
        if event_type == "tool_start":
            print(f"\n[TOOL] {data}")
        elif event_type == "token":
            print(data, end="", flush=True)
    print()

if __name__ == "__main__":
    asyncio.run(_dev_run())




# # ── User profile (simulates a filled form) ──────────────────────────────────
# # In production: loaded from Postgres by user ID.
# # In the Streamlit UI: populated from st.form() inputs.
# profile = UserProfile(
#     horizon="1-2 years",
#     risk_level="Moderate",
#     strategy="Growth",
#     markets="US (NYSE/NASDAQ)",
#     currency="USD",
#     sectors="AI & Machine Learning, SaaS, Energy",
#     max_allocation_per_stock=15,
#     min_positions=10,
#     max_positions=25,
#     rebalance_frequency="Quarterly",
#     benchmark="S&P 500",
#     esg_filter=False,
#     leverage_allowed=False,
#     custom_notes="",
# )

# # ── Build chain — profile is baked in, only {question} remains ──────────────
# chain = strategy_template.partial(**profile.model_dump()) | model | StrOutputParser()

# # ── Ask a question ───────────────────────────────────────────────────────────
# for chunk in chain.stream({"question": "What would be a good stock to buy right now?"}):
#     print(chunk, end="", flush=True)

# print()



